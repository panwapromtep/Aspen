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


class Refrig3Drum3Comp(AspenSim):
    def __init__(self, AspenFile, wdpath):
        super().__init__(AspenFile, wdpath)
    
    #overload the runSim method
    def runSim(self, x):
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
        
        sim = Simulation(AspenFileName = self.AspenFile, WorkingDirectoryPath= self.wdpath , VISIBILITY=False)
        
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
        #results = [F3, F5, F7, H3, H5, H7, H9, H11, H13]
        results = [
            sim.STRM_Get_MoleFlow("3"),
            sim.STRM_Get_MoleFlow("5"),
            sim.STRM_Get_MoleFlow("7"),
            sim.STRM_Get_Molar_Enthalpy("3"),
            sim.STRM_Get_Molar_Enthalpy("5"),
            sim.STRM_Get_Molar_Enthalpy("7"),
            sim.STRM_Get_Molar_Enthalpy("9"),
            sim.STRM_Get_Molar_Enthalpy("11"),
            sim.STRM_Get_Molar_Enthalpy("13")
            ]
        
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
        
        #tries to deal with the failed simulation gracefully
        if len(results) == 9: 
            F3, F5, F7, H3, H5, H7, H9, H11, H13 = results
        else:
            return 20000000
        
        print(F3)
        print(F5)
        print(F7)
        print(H3)
        print(H9)
        print(H11)
        print(H5)
        print(H13)
        
        # Add your cost calculation code here
        cost = F3 * ((H9 - H3)/0.65) +  F5 * ((H11 - H5)/0.65) +  F7 * ((H13 - H7)/0.65)
        return cost
    
def main():
    #! This is the start of the "driver"

    print("ok here's a test drive of the AspenSim class")
    assSim = Refrig3Drum3Comp(AspenFile = "FlashOperation.bkp", wdpath = "../FlashOperation")

    x = {
        "Flash2": {"FLASH1": [48.9, 20.7], "FLASH2": [10, 12.4], "FLASH3": [-28.9, 7.4]},
        "Heater": {"COOLER1": [4.4, 22.1], "COOLER2": [15.6, 36.5], "COOLER3": [26.7, 60.7]},
        "Compr": {"COMP1": [22.3], "COMP2": [37.2], "COMP3": [62.1]}
        }

    assSim.run_obj(x)

if __name__ == "__main__":
    main()




