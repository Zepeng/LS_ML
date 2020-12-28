import torch
from torch import nn
import pickle
import os
import junodata, model, model_identity, s2net, model_meshcnn1, model_meshcnn2
import numpy as np
from DSNBDataset_s2 import PMTIDMap, interp_pmt2mesh, GetOneEventImage, PlotRawSignal, PlotIntepSignal, save2npz
import ROOT
from collections import Counter
from tqdm import tqdm
from train import flatten_batch
import matplotlib.pyplot as plt

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def LoadModel(file_model: str):
    net = model_meshcnn1.Model(mesh_folder="/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/mesh_files/", nclasses=2)

    print('==> Resuming from checkpoint..')
    if device == 'cuda':
        checkpoint = torch.load(file_model)
    else:
        checkpoint = torch.load(file_model, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    return net


def GetPmtMap(mapfile: str):
    max_n_points_grid: bool = True
    do_calcgrid: bool = False
    pmtmap = PMTIDMap(mapfile)
    # pmtmap.CalcDict()
    n_grid = 128
    if max_n_points_grid:
        if do_calcgrid:
            pmtmap.CalcThetaPhiGrid()
        else:
            pmtmap.CalcThetaPhiPmtPoints()
    else:
        pmtmap.CalcThetaPhiSquareGrid(n_grid)
    return pmtmap


# def GetOneEventImage(pmtids:np.ndarray, hittime:np.ndarray, npes:np.ndarray, pmtmap:PMTIDMap, V,
#                      do_calcgrid:str=False, max_n_points_grid:str=True):
#     if do_calcgrid == False:
#         event2dimg = np.zeros((2, len(pmtmap.thetas)), dtype=np.float16)
#     else:
#         event2dimg = np.zeros((2, len(pmtmap.thetas), len(pmtmap.phis)), dtype=np.float16)
#
#     # event2dimg = np.zeros((2, n_grid, n_grid), dtype=np.float16)
#     event2dimg_interp = np.zeros((2, len(V)), dtype=np.float32)
#     for j in range(len(pmtids)):
#         if pmtids[j] > 17612:
#             continue
#         if max_n_points_grid:
#             if do_calcgrid:
#                 (xbin, ybin) = pmtmap.CalcBin_ThetaPhiImage(pmtids[j])
#             else:
#                 i_pmt = pmtids[j]
#         else:
#             (xbin, ybin) = pmtmap.CalcBin(pmtids[j])
#         # if ybin>124:
#         #     print(pmtids[i][j])
#         if do_calcgrid == False:
#             event2dimg[0, i_pmt] += npes[j]
#             # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
#             if event2dimg[1, i_pmt] == 0:
#                 event2dimg[1, i_pmt] = hittime[j]
#             else:
#                 event2dimg[1, i_pmt] = min(hittime[j], event2dimg[1, i_pmt])
#         else:
#             event2dimg[0, xbin, ybin] += npes[j]
#             # if event2dimg[1, xbin, ybin] < 0.001 and event2dimg[1, xbin, ybin]>-0.001:
#             if event2dimg[1, xbin, ybin] == 0:
#                 event2dimg[1, xbin, ybin] = hittime[j]
#             else:
#                 event2dimg[1, xbin, ybin] = min(hittime[j], event2dimg[1, xbin, ybin])
#
#     event2dimg_interp[0] = interp_pmt2mesh(event2dimg[0], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="linear")
#     event2dimg_interp[1] = interp_pmt2mesh(event2dimg[1], pmtmap.thetas, pmtmap.phis, V, pmtmap, method="nearest")
#     return (event2dimg, event2dimg_interp)

def Loadnpz(filename: str):
    batch = np.load(filename)
    size_predict = 100
    batchsize = size_predict
    pmtinfos = batch['pmtinfo'][:batchsize]  # [idx % self.nevt_file]
    types = batch['eventtype'][:batchsize]  # [idx % self.nevt_file]
    eqens = batch['eqen'][:batchsize]
    return (pmtinfos, types, eqens)


def RawDataPredictResult(rawfilechain: ROOT.TChain, pmtmap: PMTIDMap, net, V, name_tpyes: str = " ", name_file_train="./0.npz"):
    global __n_RawDataPredictEvt__
    CheckWhetherEqual:bool = False
    # print(__n_RawDataPredictEvt__)
    OneByOne = False
    plot_result = False
    ChangeRNRatio = False
    SingleEvetInput = False
    max_n_points_grid: bool = True
    do_calcgrid: bool = False

    if plot_result:
        if max_n_points_grid:
            if do_calcgrid:
                PHIS, THETAS = np.meshgrid(pmtmap.phis,
                                           pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            else:
                PHIS, THETAS = pmtmap.phis, pmtmap.thetas
        else:
            PHIS, THETAS = np.meshgrid(pmtmap.phis,
                                       pmtmap.thetas)  # Attention !!! Here we must be aware of the order of two inputs!!
            # print(f"thetas:{pmtmap.thetas}")
            # print(f"grid(thetas): {THETAS}")
        x_raw_grid = np.cos(THETAS) * np.cos(PHIS)
        y_raw_grid = np.cos(THETAS) * np.sin(PHIS)
        z_raw_grid = np.sin(THETAS)
        x_V, y_V, z_V = V[:, 0], V[:, 1], V[:, 2]

    fsoftmax = nn.Softmax(dim=0)
    results = []
    pmtinfos = []
    types = []
    vertices = []
    eqen_batch = []
    if name_tpyes == "Signal":
        type = 1
    else:
        type = 0
    with torch.no_grad():
        # for i in range(rawfilechain.GetEntries()):
        for i in tqdm(range(10)):
            # if i %10 ==0:
            #     print("Processing Entry: ", i)
            rawfilechain.GetEntry(i)
            pmtids = np.array(rawfilechain.PMTID, dtype=np.int32)
            npes = np.array(rawfilechain.Charge, dtype=np.float32)
            hittime = np.array(rawfilechain.Time, dtype=np.float32)
            eqen = rawfilechain.Eqen
            # print("pmtids:   ",pmtids[:10])
            # print("npes:     ",npes[:10])
            # print("hittime:  ",hittime[:10])
            # print("eqen:     ", eqen)
            # exit()
            (event2dimg, event2dimg_interp) = GetOneEventImage(pmtids, hittime, npes, pmtmap, V)
            pmtinfos.append(event2dimg_interp)
            types.append(type)
            vertices.append([rawfilechain.X, rawfilechain.Y, rawfilechain.Z])
            eqen_batch.append(eqen)
            __n_RawDataPredictEvt__ += 1
            if plot_result and __n_RawDataPredictEvt__ == 7:
                PlotRawSignal(event2dimg, x_raw_grid, y_raw_grid, z_raw_grid)
                PlotIntepSignal(event2dimg_interp, x_V, y_V, z_V)
                plt.show()
                exit()

        pmtinfos = np.array(pmtinfos)
        types = np.array(types)
        eqen_batch = np.array(eqen_batch)
        vertices = np.array(vertices)

        (pmtinfos_loadnpz, types_loadnpz, eqen_loadnpz) = Loadnpz(name_file_train)

        if CheckWhetherEqual:
            if name_tpyes == "Signal":
                pmtinfos_loadnpz = pmtinfos_loadnpz[::2]
                types_loadnpz = types_loadnpz[::2]
                print(f"check whether is {name_tpyes} :  {types[::2]}")
                print((pmtinfos[:len(pmtinfos_loadnpz)]-pmtinfos_loadnpz[:len(pmtinfos)]==0).all())
            else:
                pmtinfos_loadnpz = pmtinfos_loadnpz[1::2]
                types_loadnpz = types_loadnpz[1::2]
                print(f"check whether is {name_tpyes} :  {types[1::2]}")
                print((pmtinfos[:len(pmtinfos_loadnpz)]-pmtinfos_loadnpz[:len(pmtinfos)]==0).all())
                # exit()
        # save2npz("./try_save.npz", pmtinfos, types, eqen_batch, vertices)

        if ChangeRNRatio:
            n_0 = 30
            n_1 = 30
            pmtinfos_input = np.array(pmtinfos_loadnpz[:n_0:2])
            print("Before append:   ",pmtinfos_input.shape)
            pmtinfos_input = np.concatenate((pmtinfos_input, pmtinfos_loadnpz[1:n_1:2]))
            print("After append:   ",pmtinfos_input.shape)
            types_input = types_loadnpz[:n_0:2]
            types_input = np.append(types_input, types_loadnpz[1:n_1:2])
            # print(types_input)
        else:
            pmtinfos_input = pmtinfos_loadnpz
            types_input = types_loadnpz

        if SingleEvetInput:
            print("Before put into single:  ",pmtinfos_input.shape)
            pmtinfos_input = pmtinfos_input[0].reshape(1, 2, 10242)
            print("After put into single:   ",pmtinfos_input.shape)
            types_input = types_input[0]

        if OneByOne:
            PredictEventOneByOne(pmtinfos_input, types_input)
        else:
            PredictEventOneBatch(pmtinfos_input, types_input)
        # inputs = torch.from_numpy(np.array(pmtinfos_input))
        # inputs  = inputs.to(device)
        # outputs = net(inputs)
        # outputs = fsoftmax(outputs)
        # print("output: ", outputs)
        # _, predicted = outputs.max(0)
        # # correct = predicted.eq(torch.from_numpy(types_input)).sum().item()
        # results= predicted.numpy()
        # print(name_tpyes," Prediction :  ",results, ",   Answer :",types_input)
        # print(name_tpyes," :  ",Counter(np.array(results)))

        # print("Score: ", correct/len(types_input))

def PredictEventOneByOne(pmtinfos, types):
    fsoftmax = nn.Softmax(dim=0)
    ar_shape = np.asarray(pmtinfos.shape)
    ar_shape = np.insert(ar_shape[1:], 0, 1)
    predictions = []
    with torch.no_grad():
        for i in range(len(pmtinfos)):
            pmtinfos_input = pmtinfos[i].reshape(tuple(ar_shape))
            types_input = types[i]
            inputs = torch.from_numpy(np.array(pmtinfos_input))
            inputs  = inputs.to(device)
            outputs = net(inputs)
            outputs = fsoftmax(outputs)
            print("output: ", outputs)
            _, predicted = outputs.max(0)
            # correct = predicted.eq(torch.from_numpy(types_input)).sum().item()
            results= predicted.item()
            predictions.append(results)
            print(" Prediction :  ",results, ",   Answer :",types_input)
    print("Prediction : ", predictions, "  , Answer :  ", types)
    print(Counter(predictions==types))
def PredictEventOneBatch(pmtinfos, types, name_types=""):
    fsoftmax = nn.Softmax(dim=1)
    inputs = torch.from_numpy(np.array(pmtinfos))
    inputs  = inputs.to(device)
    outputs = net(inputs)
    outputs = fsoftmax(outputs)
    print("output: ", outputs)
    _, predicted = outputs.max(1)
    correct = predicted.eq(torch.from_numpy(types)).sum().item()
    results= predicted.numpy()
    print(name_types," Prediction :  ",results, ",   Answer :",types)
    # print(name_tpyes," :  ",Counter(np.array(results)))

    print("Score: ", correct/len(types))
def npzPredictResult(filename: str, net):
    with torch.no_grad():
        fsoftmax = nn.Softmax(dim=1)
        batch = np.load(filename)
        size_predict = 100
        batchsize = size_predict
        pmtinfos = batch['pmtinfo'][:batchsize]  # [idx % self.nevt_file]
        types = batch['eventtype'][:batchsize]  # [idx % self.nevt_file]
        eqens = batch['eqen'][:batchsize]
        # pmtinfos = batch['pmtinfo'][200:200+batchsize] #[idx % self.nevt_file]
        # types = batch['eventtype'][200:200+batchsize] #[idx % self.nevt_file]
        # eqens    = batch['eqen'][200:200+batchsize]
        # for i in tqdm(range(size_predict)):
        inputs = torch.from_numpy(np.array(pmtinfos))
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = fsoftmax(outputs)
        _, predicted = outputs.max(1)
        correct = predicted.eq(torch.from_numpy(types)).sum().item()
        results = predicted.numpy()
        print("Predict Result:  ", results)
        print("Anwser:   ", types[:size_predict])
        print(f"Correct Ratio: {correct / len(types)} ({correct}/{len(types)})")


if __name__ == "__main__":
    __n_RawDataPredictEvt__ = 0
    file_model = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/usgcnn_train_TimeNearest_lr0.5decay/checkpoint_sens/ckpt.t7"
    # filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/data_usgcnn_TimeNearest/0.npz"
    # filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/try_save.npz"
    filename = "/afs/ihep.ac.cn/users/l/luoxj/s2cnn_classifier/0_noShuffle.npz"
    file_mesh = "/afs/ihep.ac.cn/users/l/luoxj/gpu_500G/ugscnn/mesh_files/icosphere_5.pkl"
    mapfile = "/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv"
    rawfile_bkg = "/workfs/exo/zepengli94/JUNO_DSNB/AtmNu/data/atm_000001.root"
    rawfile_sig = "/workfs/exo/zepengli94/JUNO_DSNB/DSNB/data/dsnb_000001.root"
    p = pickle.load(open(file_mesh, "rb"))
    V = p['V']
    print("Loading Model")
    net = LoadModel(file_model)

    pmtmap = GetPmtMap(mapfile)
    rawfilechain_bkg = ROOT.TChain('psdtree')
    # sigchain.Add('%s/*00001.root' % sig_dir)
    rawfilechain_bkg.Add(rawfile_bkg)
    rawfilechain_sig = ROOT.TChain('psdtree')
    # sigchain.Add('%s/*00001.root' % sig_dir)
    rawfilechain_sig.Add(rawfile_sig)
    # RawDataPredictResult(rawfilechain_bkg, pmtmap, net, V, "Background", filename)
    RawDataPredictResult(rawfilechain_sig, pmtmap, net, V, "Signal", filename)
    # npzPredictResult(filename, net)
