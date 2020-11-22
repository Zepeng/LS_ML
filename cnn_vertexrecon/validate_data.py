import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("/afs/ihep.ac.cn/users/l/luoxj/Style/Paper.mplstyle")
if __name__ == "__main__":
    filename = "/afs/ihep.ac.cn/users/l/luoxj/junofs_500G/data_cnnVertex/data_J20v2/sample_173.npz"
    batch = np.load(filename)
    pmtinfos = batch['pmtinfo']  # [idx % self.nevt_file]
    vertices = batch['vertex']  # [idx % self.nevt_file]
    edeps = batch['edep']
    print("n_events   :",len(pmtinfos))
    with PdfPages("./output_pdf/"+filename.split('/')[-1].split('.')[0]+"_OneEvtHeatMap.pdf") as pdf:
        for i in range(5):
            fig_pmtinfos = plt.figure( figsize=(10,5) )
            ax1_pmtinfos =  fig_pmtinfos.add_subplot(1, 2, 1)
            sns.heatmap(pmtinfos[i][0])
            plt.xlabel("Theta Bin")
            plt.ylabel("Phi Bin")
            plt.title("Edep [ PE ]")
            ax2_pmtinfos = fig_pmtinfos.add_subplot(1, 2, 2)
            sns.heatmap(pmtinfos[i][1] )
            plt.xlabel("Theta Bin")
            plt.ylabel("Phi Bin")
            plt.title("Hit Time [ ns ]")
            pdf.savefig()
    plt.show()



