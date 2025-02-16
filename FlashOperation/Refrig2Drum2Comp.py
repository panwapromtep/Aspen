# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 12:59:01 2025

@author: wsangpa1
"""

import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)


from AspenSim import AspenSim
from CodeLibrary import Simulation


class Refrig2Drum2Comp(AspenSim):
    def __init__(self, AspenFile, wdpath):
        super().__init__(AspenFile, wdpath)
    
    #overload the runSim method
    def runSim(self, x, visibility = False):
        """
        Runs the refrigeration simulation with the given parameters.

        Args:            
            x (dictionary): Categorizes different types of process blocks (e.g., FLASH2, HEATER) 
            and maps them to their names with an associated parameter list
            
            Structure:
                x = {
                    "BLOCK_TYPE": {"BLOCK_NAME": [PARAM1, PARAM2]}
                    "FLASH2": {"BLOCK_NAME": [temp, pressure]}
                    "HEATER": {"BLOCK_NAME": [temp, pressure]}
                    "COMPR": {"BLOCK_NAME": [discharge_pressure]}
                }
        Returns:
            obj: The result of the simulation.
        """
        
        sim = Simulation(AspenFileName = self.AspenFile, WorkingDirectoryPath= self.wdpath , VISIBILITY=visibility)
        
        #loop through flashdrums and set values
        for blockname in x["Flash2"]:
            sim.BLK_FLASH2_Set_Temperature(blockname, x["Flash2"][blockname][0])
            sim.BLK_FLASH2_Set_Pressure(blockname, x["Flash2"][blockname][1])
            
        #loop thorugh heaters and set values
        for blockname in x["Heater"]:
            sim.BLK_HEATER_Set_Temperature(blockname, x["Heater"][blockname][0])
            sim.BLK_HEATER_Set_Pressure(blockname, x["Heater"][blockname][1])
            
        #loop thorugh compressors and set values
        for blockname in x["Compr"]:
            sim.BLK_COMPR_Set_Discharge_Pressure(blockname, x["Compr"][blockname][0])
            
            
        #run the simulation
        sim.DialogSuppression(TrueOrFalse= True)
        sim.Run()
    
        #compile the results
        #results = [F3, F5, H3, H5, H9, H11]
        results = {
            "OUTF1_MF": sim.STRM_Get_MoleFlow("OUTF1"),
            "OUTF2_MF": sim.STRM_Get_MoleFlow("OUTF2"),
            "OUTF1_H": sim.STRM_Get_Molar_Enthalpy("OUTF1"),
            "OUTF2_H": sim.STRM_Get_Molar_Enthalpy("OUTF2"),
            "OUTCOMP1_H": sim.STRM_Get_Molar_Enthalpy("OUTCOMP1"),
            "OUTCOMP2_H": sim.STRM_Get_Molar_Enthalpy("OUTCOMP2"),
        }
        
        #print(len(results))
        #print(results)
        
        sim.SaveAs("FlashOperation_super_test.bkp", True)
        sim.CloseAspen()
        
        return results

    # overloading the costFunc method
    def costFunc(self, results):
        """
        Calculates the cost based on the simulation results.

        """
            
        OUTCOMP1 = results["OUTCOMP1_H"]
        OUTF1 = results["OUTF1_H"]
        OUTCOMP2 = results["OUTCOMP2_H"]
        OUTF2 = results["OUTF2_H"]
        FLOW_1 = results["OUTF1_MF"]
        FLOW_2 = results["OUTF2_MF"]
        
        if FLOW_1 == 0:
            OUTF1 = 0
            OUTCOMP1 = 0
        if FLOW_2 == 0:
            OUTF2 = 0
            OUTCOMP2 = 0
            
        
        # Add your cost calculation code here
        cost = FLOW_1 * ((OUTCOMP1 - OUTF1)/0.65) +  FLOW_2 * ((OUTCOMP2 - OUTF2)/0.65) 
        return cost
    
def main():
    #! This is the start of the "driver"

    print("ok here's a test drive of the AspenSim class")
    assSim = Refrig2Drum2Comp(AspenFile = "FlashOperation.bkp", wdpath = "../FlashOperation")

    x = {
        "Flash2": {"FLASH1": [48.9, 20.7], "FLASH2": [10, 12.4]},
        "Heater": {"COOLER1": [4.4, 22.1], "COOLER2": [15.6, 36.5]},
        "Compr": {"COMP1": [22.3], "COMP2": [37.2]}
        }

    res = assSim.run_obj(x, False)
    print(res)

if __name__ == "__main__":
    main()




