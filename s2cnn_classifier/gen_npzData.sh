#!/bin/bash
sig_dir=/workfs/exo/zepengli94/JUNO_DSNB/DSNB/data/
bkg_dir=/workfs/exo/zepengli94/JUNO_DSNB/AtmNu/data/
original_dir=`pwd`
gen_list=0
if [ ! -f sig_file.list ] || [ ! -f bkg_file.list ] || [[ gen_list == 1 ]]
then
    pushd $sig_dir
    ls *.root | tr " " "\n" > $original_dir/sig_file.list
    popd
    pushd $bkg_dir
    ls *.root | tr " " "\n" > $original_dir/bkg_file.list
    popd
fi
i=0
exec 3<"sig_file.list"
exec 4<"bkg_file.list"
entries_sig=1000
entries_bkg=330 #this is the max entries of bkg_files used to be set the start point of jobs
n_step=$(($entries_sig/entries_bkg))
echo "n_step:  $n_step"
# while read sig_file <&3 && read bkg_file <&4
while read sig_file <&3
do
    for j in `seq 1 $n_step`
    do
        read bkg_file<&4
        echo "source /afs/ihep.ac.cn/users/l/luoxj/junofs_500G/anaconda3/etc/profile.d/conda.sh && conda activate tf &&"> ./jobs_DSNB/jobs_$i.sh
        echo "time python DSNBDataset_s2.py -s /workfs/exo/zepengli94/JUNO_DSNB/DSNB/data/$sig_file -b /workfs/exo/zepengli94/JUNO_DSNB/AtmNu/data/$bkg_file -o ./data/$i.npz --pmtmap /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv -e $((($j-1)*$entries_bkg))" >> ./jobs_DSNB/jobs_$i.sh
        i=$(($i+1))
#         echo $bkg_file
    done
done
chmod 755 ./jobs_DSNB/*.sh




