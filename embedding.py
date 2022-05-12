import torch
import numpy as np
from tqdm.auto import tqdm
# from tqdm import tqdm_notebook as tqdm
from sklearn.covariance import LedoitWolf

from sklearn.cluster import KMeans
def _kmeans_fun_sklearn(X, K=10):
    _X = X.detach().cpu().numpy()
    D = _X.shape[1]
    _kmeans = KMeans(n_clusters=K, max_iter=1000, verbose=0, tol=1e-40)
    _kmeans.fit(_X)
    # logger.info(_kmeans.cluster_centers_)

    k_men = _kmeans.cluster_centers_
    k_var = np.zeros([K, D, D])

    _dist = euclidean_metric_np(_X, k_men)
    _idx_min = np.argmin(_dist, axis=1)
    for k in range(K):
        samples = _X[k == _idx_min]
        _m = np.mean(samples, axis=0)
        k_var[k] = LedoitWolf().fit(samples).covariance_

    return k_men, k_var

def euclidean_metric_np(X, centroids):
    X = np.expand_dims(X, 1)
    centroids = np.expand_dims(centroids, 0)
    dists = (X - centroids) ** 2
    dists = np.sum(dists, axis=2)
    return dists


def euclidean_metric_gpu(X, centers):
    X = X.unsqueeze(1)
    centers = centers.unsqueeze(0)

    dist = torch.sum((X - centers) ** 2, dim=-1)
    return dist


def _kmeans_fun_gpu(X, K=10, max_iter=1000, batch_size=8096, tol=1e-1):
    N = X.shape[0]
    D = X.shape[1]

    indices = torch.randperm(N)[:K]
    init_centers = X[indices]

    batchs = N // batch_size
    last = 1 if N % batch_size != 0 else 0

    choice_cluster = torch.zeros([N]).cuda()
    for _ in range(max_iter):
        for bn in range(batchs + last):
            if bn == batchs and last == 1:
                _end = -1
            else:
                _end = (bn + 1) * batch_size
            X_batch = X[bn * batch_size: _end]

            dis_batch = euclidean_metric_gpu(X_batch, init_centers)
            choice_cluster[bn * batch_size: _end] = torch.argmin(dis_batch, dim=1)

        init_centers_pre = init_centers.clone()
        for index in range(K):
            selected = torch.nonzero(choice_cluster == index).squeeze().cuda()
            init_centers[index] = torch.index_select(X, 0, selected).mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((init_centers - init_centers_pre) ** 2, dim=1)
            ))
        if center_shift < tol:
            break

    k_men = init_centers.detach().cpu().numpy()
    k_var = np.zeros([K, D, D])
    _X = X.detach().cpu().numpy()
    _dist = euclidean_metric_np(_X, k_men)
    _idx_min = np.argmin(_dist, axis=1)
    for k in range(K):
        samples = _X[k == _idx_min]
        _m = np.mean(samples, axis=0)
        k_var[k] = LedoitWolf().fit(samples).covariance_
    torch.cuda.empty_cache()
    return k_men, k_var


def _cal_var(X, centers=None, choice_cluster=None, K=10):
    D = X.shape[1]
    k_var = np.zeros([K, D, D])
    eps = np.eye(D) * 1e-10
    if centers is not None:
        _dist = euclidean_metric_np(X, centers)
        choice_cluster = np.argmin(_dist, axis=1)

    for k in range(K):
        samples = X[k == choice_cluster]
        _m = np.mean(samples, axis=0)
        k_var[k] = LedoitWolf().fit(samples).covariance_ + eps
    return k_var.astype(np.float32)


def _cal_var_gpu(X, centers=None, choice_cluster=None, K=10):
    D = X.shape[1]
    k_var = torch.zeros([K, D, D]).cuda()
    if centers is not None:
        _dist = euclidean_metric_gpu(X, centers)
        choice_cluster = torch.argmin(_dist, dim=1)
    for k in range(K):
        samples = X[k == choice_cluster]
        _m = torch.mean(samples, dim=0)
        k_var[k] = torch.mm(X.t(), X) / (X.shape[0] - 1)
    return k_var

def mahalanobias_metric_gpu(X, mean, var):
    torch.cuda.empty_cache()
    dis = torch.zeros([X.shape[0], mean.shape[0]])
    for k in range(mean.shape[0]):
        _m = mean[k]
        _inv = torch.inverse(var[k])
        # method 1
        delta = X - _m
        temp = torch.mm(delta, _inv)
        dis[:, k] = torch.sqrt_(torch.sum(torch.mul(delta, temp), dim=1))
    return dis

def _kgaussians_fun_gpu(X, K=10, max_iter=1000, batch_size=8096, tol=0.1):
    N = X.shape[0]

    _X = X.detach().cpu().numpy()
    print('Init centers...')
    # indices = torch.randperm(N)[:K]
    # init_centers = X[indices]
    init_centers, _ = _kmeans_fun_gpu(X, K=K, max_iter=1000, tol=tol)
    # init_centers, _ = _kmeans_fun_sklearn(X, K=K)
    init_centers = torch.from_numpy(init_centers.astype(np.float32)).cuda()
    print('Init centers done...')
    k_var = _cal_var(_X, centers=init_centers.detach().cpu().numpy(), K=K)
    k_var = torch.from_numpy(k_var).cuda()

    __k_var = k_var.clone()
    __init_centers = init_centers.clone()
    try:
    # if True:
        batchs = N // batch_size
        last = 1 if N % batch_size != 0 else 0

        choice_cluster = torch.zeros([N]).cuda()
        # pre_choice_cluster = choice_cluster.clone()
        for i in tqdm(range(max_iter), desc="[Gaussian Clustering]",ascii=True, position=0, leave=True, ncols=80):
            torch.cuda.empty_cache()
            # print(f"KGaussians iteration {i+1}...")
            # init_choice_cluster = choice_cluster.clone()
            for bn in range(batchs + last):
                if bn == batchs and last == 1:
                    _end = -1
                else:
                    _end = (bn + 1) * batch_size
                X_batch = X[bn * batch_size: _end]

                dis_batch = mahalanobias_metric_gpu(X_batch, init_centers, k_var)
                choice_cluster[bn * batch_size: _end] = torch.argmin(dis_batch, dim=1)

            init_centers_pre = init_centers.clone()
            for index in range(K):
                selected = torch.nonzero(choice_cluster == index).squeeze().cuda()
                init_centers[index] = torch.index_select(X, 0, selected).mean(dim=0)

            k_var_pre = k_var.clone()
            k_var = _cal_var(_X, choice_cluster=choice_cluster.detach().cpu().numpy(), K=K)
            k_var = torch.from_numpy(k_var).cuda()
            center_shift = torch.mean(
                torch.sqrt(
                    torch.sum((init_centers - init_centers_pre) ** 2, dim=1)
                ))
            var_shift = torch.mean(
                torch.sqrt(
                    torch.sum((k_var - k_var_pre) ** 2, dim=1)
                ))
            print(f"center and var shift: {center_shift.item():.12f}, {var_shift.item():.12f}, "
                  f"{(center_shift + var_shift).item():.12f}")
            str = ""
            for i in range(K):
                str += f"{i}-{torch.sum(choice_cluster==i).item()}, "
            print(str)
            if (center_shift + var_shift) < tol:
                break 

        k_men = init_centers.detach().cpu().numpy()
        k_var = k_var.detach().cpu().numpy()
        torch.cuda.empty_cache()
        return k_men, k_var
    except:
    # else:
        print(f"Init again...")
        k_men, k_var = _kmeans_fun_gpu(X, K=K, max_iter=1000, tol=tol*0.1)
        print(f"Init again done...")
        torch.cuda.empty_cache()
        return k_men, k_var

