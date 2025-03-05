# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:46 2025

@author: ppromte1
"""

import sys
import os
import numpy as np

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)


from AspenSim import AspenSim
from CodeLibrary import Simulation
from FlashOperation.Refrig2Drum2Comp import Refrig2Drum2Comp

#overload only the cost function to include the constraint on temperature
class Refrig2DrumConstraintHeatExConstr(Refrig2Drum2Comp):
    def __init__(self, AspenFile, wdpath, visibility = False, Penalty = 1e5):
        super().__init__(AspenFile, wdpath, visibility)
        self.open_simulation()
        self.Penalty = Penalty
        
    @staticmethod
    def flatten_params(x_dict):
        flat_list = []
        for block_type in ["Flash2"]:
            for block, params in x_dict[block_type].items():
                flat_list.extend(params)
        return np.array(flat_list)

    @staticmethod
    def unflatten_params(flat_array):
        x_dict = {
            "Flash2": {"FLASH1": [flat_array[0]], "FLASH2": [flat_array[1]]},
        }
        return x_dict
        
    def runSim(self, x):
        self.open_simulation()
        for blockname, params in x["Flash2"].items():
            #print("Setting Flash2", blockname, params)
            self.sim.BLK_FLASH2_Set_Pressure(blockname, params[0])

        self.sim.DialogSuppression(True)
        self.sim.Run()

        results = {
            "OUTF1_MF": self.sim.STRM_Get_MoleFlow("OUTF1"),
            "OUTF2_MF": self.sim.STRM_Get_MoleFlow("OUTF2"),
            "OUTF1_H": self.sim.STRM_Get_Molar_Enthalpy("OUTF1"),
            "OUTF2_H": self.sim.STRM_Get_Molar_Enthalpy("OUTF2"),
            "OUTCOMP1_H": self.sim.STRM_Get_Molar_Enthalpy("OUTCOMP1"),
            "OUTCOMP2_H": self.sim.STRM_Get_Molar_Enthalpy("OUTCOMP2"),
            "TEMPOUT": self.sim.STRM_Get_Temperature("OUTSTR")
        }
        return results
    
    

    def costFunc(self, results):
        FLOW_1 = results["OUTF1_MF"] # kmol/hr
        FLOW_2 = results["OUTF2_MF"] # kmol/hr
        OUTCOMP1 = results["OUTCOMP1_H"] # cal/mol
        OUTF1 = results["OUTF1_H"] # cal/mol
        OUTCOMP2 = results["OUTCOMP2_H"] # cal/mol
        OUTF2 = results["OUTF2_H"] # cal/mol
        TEMPOUT = results["TEMPOUT"] # Celcius
        
        Penalty = self.Penalty
        
        if FLOW_1 == 0:
            OUTF1 = OUTCOMP1 = 0
        if FLOW_2 == 0:
            OUTF2 = OUTCOMP2 = 0

        if TEMPOUT < -28.9:
            cost = (1000 * FLOW_1 * ((OUTCOMP1 - OUTF1) / 0.65) + \
                1000 * FLOW_2 * ((OUTCOMP2 - OUTF2) / 0.65))/4184 
        else:
            cost = (1000 * FLOW_1 * ((OUTCOMP1 - OUTF1) / 0.65) + \
                1000 * FLOW_2 * ((OUTCOMP2 - OUTF2) / 0.65))/4184 + \
                Penalty * (TEMPOUT + 28.9)**2
                
        print(f"TEMPOUT: {TEMPOUT}")
        return cost

    def run_obj(self, x_dict):
        results = self.runSim(x_dict)
        cost = self.costFunc(results)
        print("The cost is {:.2f}".format(cost))
        return cost

