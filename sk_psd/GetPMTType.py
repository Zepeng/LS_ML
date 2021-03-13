# -*- coding:utf-8 -*-
# @Time: 2021/1/28 16:40
# @Author: Luo Xiaojie
# @Email: luoxj@ihep.ac.cn
# @File: GetPMTType.py
import numpy as np

class PMTType:
    def __init__(self):
        self.check = False
        self.list_HAM = np.loadtxt("/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J20v2r0-Pre0/data/Simulation/ElecSim/Hamamatsu_pmtID.txt", dtype=np.int32)
        self.v_flag_HAM = np.zeros(17613, dtype=np.int32)
        self.v_flag_HAM[self.list_HAM] = 1
        if self.check:
            from collections import Counter
            print(self.v_flag_HAM)
            print(Counter(self.v_flag_HAM))
            print(f"n of HAM : {len(self.list_HAM)}")

    def GetPMTType(self,pmtid:int):
        """

        Args:
            pmtid: to get pmt type

        Returns:
            1 or 0, 1 means this pmtid map to HAM , 0 means this pmtid map to MCP

        """
        if self.v_flag_HAM[pmtid] ==1:
            return True
        else:
            return False
if __name__ == '__main__':
    pmt_type = PMTType()
    print(pmt_type.GetPMTType(11))