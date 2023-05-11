from multiprocessing import Process
import multiprocessing
import pickle
import numpy as np
# from sklearn.cluster import KMeans
from utils.kmeans import KMeans


def gen_universal_set(s_size=4):
    base_p1 = [1, 2**(-1), 2**(-5), 0]
    base_p2 = [1, 2**(-2), 2**(-6), 0]
    base_p3 = [1, 2**(-3), 2**(-7), 0]
    base_p4 = [1, 2**(-4), 2**(-8), 0]
    uset_p = []
    for i in range(s_size):
        for j in range(s_size):
            for k in range(s_size):
                for l in range(s_size):
                    pv = (base_p1[i] + base_p2[j] + base_p3[k] + base_p4[l]) / 4
                    ap_flag = True
                    for t in range(len(uset_p)):
                        if pv == uset_p[t]:
                            ap_flag = False
                    if ap_flag:
                        uset_p.append(pv)
                        uset_p.append(-pv)
    return uset_p


def solve(name, model_name):
    print('cluster {} module'.format(name))

    path = "data/{}/{}.pkl".format(model_name, name)
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)

    x = data['raw_input']
    module_name = data['name']
    postReLU = data['postReLU']
    bit = data['bit']
    d = x.shape 

    mu = np.mean(x, axis=(2, 3), keepdims=True)
    maxv = np.max(np.abs(x), axis=(2, 3), keepdims=True)

    if postReLU == True:
        x = x / (maxv + 1e-8)
    else:
        x = (x - mu)
        maxv = np.max(np.abs(x), axis=(2, 3), keepdims=True)
        x = x / (maxv + 1e-8)

    x = x.transpose((1, 0, 2, 3)).reshape(d[1], -1)  
    num_clusters = 2 ** bit
    iters = 5
    su = gen_universal_set()
    print('PostReLu: {} universal set: {} (size: {})'.format(postReLU, su, len(su)))
    su = np.array(list(set(su)), dtype=float).reshape(1, -1)  
    
    log_folder_path = "log/{}".format(model_name)
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    txt_path = "log/{}/{}_maxv.txt".format(model_name,
                                           module_name.replace('.', '_'))
    dc = {}
    with open(txt_path, mode='w', encoding='utf-8') as f:
        for channel in range(0, x.shape[0]):
            dc[channel] = {}
            prev_loss = 100000000
            first_loss = 100000000
            best_qps = None
            best_centers = None
            f.write("========Channel: {}=========\n".format(channel))
            print("========Channel: {}=========\n".format(channel))
            x_ = x[channel, :].reshape(-1, 1)  # [b*h*w,1]
            for iter in range(iters):
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(x_)
                clusters = kmeans.centers

                qps = su[0, np.argmin(np.abs(su - clusters), 1)
                         ].reshape(num_clusters, 1)  # [16,1]
                center_loss = np.sum(np.abs(qps - clusters))
                quant_loss = np.sum(np.min(np.abs(
                    (x_ - np.reshape(qps, newshape=(1, num_clusters)))), axis=1)**2)  # [b*h*w, 1] - [1,16]
                inertia = np.sum(np.min(
                    np.abs((x_ - np.reshape(clusters, newshape=(1, num_clusters)))), axis=1)**2)

                f.write("iter: {}, cluster sse:{:.3f}, center_loss:{:.3f}, quant_loss:{:.3f}\n".format(
                    iter, inertia, center_loss, quant_loss))

                print("iter: {}, cluster sse:{:.3f}, center_loss:{:.3f}, quant_loss:{:.3f}\n".format(
                    iter, inertia, center_loss, quant_loss))

                f.write("qps: {}\n, clusters:{}\n".format(qps, clusters))
                print("qps: {}\n, clusters:{}\n".format(qps, clusters))

                if prev_loss > quant_loss:
                    prev_loss = quant_loss
                    best_qps = qps

                if first_loss > inertia:
                    first_loss = inertia
                    best_centers = clusters

            dc[channel]['qps'] = best_qps
            dc[channel]['quant_loss'] = prev_loss
            dc[channel]['centers'] = best_centers
            dc[channel]['sse_loss'] = first_loss

            f.write("Channel: {}, best qps: {}\n, quant loss:{:.3f}\n, centers:{}, first_loss:{:.3f}".format(
                channel, best_qps, prev_loss, best_centers, first_loss))
            print("Channel: {}, best qps: {}\n, quant loss:{:.3f}\n, centers:{}, first_loss:{:.3f}".format(
                channel, best_qps, prev_loss, best_centers, first_loss))

    params_folder_path = "result/{}".format(model_name)
    if not os.path.exists(params_folder_path):
        os.makedirs(params_folder_path)
    
    params_path = "{}/{}_maxv.pkl".format(params_folder_path,
                                          module_name.replace('.', '_'))
    with open(params_path, mode='wb') as f:
        pickle.dump(dc, f)


if __name__ == '__main__':
    import os
    import time
    process_list = []
    model_name = "edsr_4x_4bit" # if you change the model or bit, it need be modified
    path = "data/"+model_name
    file = os.listdir(path)
    
    start_time = time.time()
    pool = multiprocessing.Pool(processes=20)  
    results = []
    for i in range(0, len(file)):
        results.append(pool.apply_async(solve, args=(
            file[i].split(".pkl")[0], model_name)))  

    pool.close()
    pool.join()
    end_time = time.time()
    elapsed_time = end_time - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"程序运行时间：{hours:02d}:{minutes:02d}:{seconds:02d}")
