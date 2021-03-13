import numpy as np
import matplotlib.pyplot as plt
def LoadnpzDict(name_file:str):
    dataset = np.load(name_file, allow_pickle=True)
    eff_sig_dir = dataset["eff_sig"].item()
    eff_bkg_dir = dataset["eff_bkg"].item()
    return (eff_sig_dir, eff_bkg_dir)
def PlotResult(input_filelist:list):
    fig, ax = plt.subplots()
    for file in input_filelist:
        if "No" in file:
            type_line:str = "--"
            label_file = "(w/o charge)"
        elif "Combine" in file:
            type_line:str = ":"
            label_file = "(combine)"
        else:
            type_line:str = "-"
            label_file = "(w/ charge)"
        (eff_sig, eff_bkg) = LoadnpzDict(file)
        eff_sig_ave = np.array([])
        for i, key in enumerate(eff_sig.keys()):
            ax.plot(eff_bkg[key], eff_sig[key], type_line, label=f"${key.replace('$', '')}m^3${label_file}")
            if key == '$R^3$<4096':
                continue
            elif len(eff_sig_ave) == 0:
                eff_sig_ave = np.array(eff_sig[key])
            else:
                eff_sig_ave += np.array(eff_sig[key])
        eff_sig_ave /= 4.
        ax.plot(eff_bkg[list(eff_bkg.keys())[0]], eff_sig_ave, type_line, label=f"average 4 volumes{label_file}")
    ax.plot([0.01, 0.01], [0, 1], "-", linewidth=2, color="cornflowerblue" )
    ax.set_xlim(0, 0.04)
    ax.set_ylim(0.55, 1)
    plt.xlabel("Remaining background efficiency")
    plt.ylabel("Signal efficiency")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    n_plot = 2
    dir_model_prefix = "model_maxtime_"
    # dir_model_suffix = "_jobs_DSNB_sk_data/"
    dir_model_suffix = "_job_data_dividemax/"
    v_data_type = ["time", "WeightEtime", "combine"][:n_plot]
    input_filelist = ["eff_timeNoWeightE_input.npz", "eff_timeWeightE_input.npz", "eff_timeCombine_input.npz"][:n_plot]
    for i in range(len(input_filelist)):
        input_filelist[i] = dir_model_prefix+v_data_type[i]+dir_model_suffix+input_filelist[i]
    print(input_filelist)
    # input_filelist = ["eff_timeCombine_input.npz"]
    # input_filelist = ["eff_timeNoWeightE_input.npz", "eff_timeWeightE_input.npz"]
    PlotResult(input_filelist)