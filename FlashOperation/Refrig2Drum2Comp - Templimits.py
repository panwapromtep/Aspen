# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:59:01 2025

@author: wsangpa1
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


class Refrig2Drum2Comp(AspenSim):
    def __init__(self, AspenFile, wdpath, visibility = False):
        super().__init__(AspenFile, wdpath, visibility)
        self.sim = None
        self.open_simulation()
        

    def reset(self):
        if self.sim is not None:
            try:
                # Ensure the method is invoked, not just referenced
                self.sim.EngineReinit()
                print("🔄 Aspen simulation successfully reinitialized.")
            except Exception as e:
                raise RuntimeError(f"Failed to reinitialize Aspen simulation: {e}")
        else:
            raise RuntimeError("Simulation object (self.sim) is None; cannot reset.")


    @staticmethod
    def flatten_params(x_dict):
        flat_list = []
        for block_type in ["Flash2", "Heater", "Compr"]:
            for block, params in x_dict[block_type].items():
                flat_list.extend(params)
        return np.array(flat_list)

    @staticmethod
    def unflatten_params(flat_array):
        x_dict = {
            "Flash2": {"FLASH1": [flat_array[0]], "FLASH2": [flat_array[1]]},
            "Heater": {"COOLER1": [flat_array[2], flat_array[3]], "COOLER2": [flat_array[4], flat_array[5]]},
            "Compr": {"COMP1": [flat_array[6]], "COMP2": [flat_array[7]]}
        }
        return x_dict
    
    def open_simulation(self):
        if self.sim is None:
            self.sim = Simulation(AspenFileName=self.AspenFile, 
                                  WorkingDirectoryPath=self.wdpath, 
                                  VISIBILITY=self.visibility
                                  )

    def close_simulation(self):
        if self.sim:
            self.sim.CloseAspen()
            self.sim = None

    def runSim(self, x):
        self.open_simulation()

        for blockname, params in x["Flash2"].items():
            print("Setting Flash2", blockname, params)
            self.sim.BLK_FLASH2_Set_Pressure(blockname, params[0])

        for blockname, params in x["Heater"].items():
            print("Setting Heater", blockname, params)
            self.sim.BLK_HEATER_Set_Temperature(blockname, params[0])
            self.sim.BLK_HEATER_Set_Pressure(blockname, params[1])

        for blockname, params in x["Compr"].items():
            print("Setting Compr", blockname, params)
            self.sim.BLK_COMPR_Set_Discharge_Pressure(blockname, params[0])

        self.sim.DialogSuppression(True)
        self.sim.Run()

        results = {
            "OUTF1_MF": self.sim.STRM_Get_MoleFlow("OUTF1"),
            "OUTF2_MF": self.sim.STRM_Get_MoleFlow("OUTF2"),
            "OUTF1_H": self.sim.STRM_Get_Molar_Enthalpy("OUTF1"),
            "OUTF2_H": self.sim.STRM_Get_Molar_Enthalpy("OUTF2"),
            "OUTCOMP1_H": self.sim.STRM_Get_Molar_Enthalpy("OUTCOMP1"),
            "OUTCOMP2_H": self.sim.STRM_Get_Molar_Enthalpy("OUTCOMP2"),
        }
        return results

    def costFunc(self, results):
        FLOW_1 = results["OUTF1_MF"]
        FLOW_2 = results["OUTF2_MF"]
        OUTCOMP1 = results["OUTCOMP1_H"]
        OUTF1 = results["OUTF1_H"]
        OUTCOMP2 = results["OUTCOMP2_H"]
        OUTF2 = results["OUTF2_H"]

        if FLOW_1 == 0:
            OUTF1 = OUTCOMP1 = 0
        if FLOW_2 == 0:
            OUTF2 = OUTCOMP2 = 0

        cost = FLOW_1 * ((OUTCOMP1 - OUTF1) / 0.65) + FLOW_2 * ((OUTCOMP2 - OUTF2) / 0.65)
        return cost
    
def main():
    #! This is the start of the "driver"

    print("ok here's a test drive of the AspenSim class")
    assSim = Refrig2Drum2Comp(AspenFile = "FlashOperation.bkp", wdpath = "../FlashOperation")

    x = {
        "Flash2": {"FLASH1": [20.7], "FLASH2":  [12.4]},
        "Heater": {"COOLER1": [4.4, 22.1], "COOLER2": [15.6, 36.5]},
        "Compr": {"COMP1": [22.3], "COMP2": [37.2]}
        }

    res = assSim.run_obj(x)
    print(res)

if __name__ == "__main__":
    main()




