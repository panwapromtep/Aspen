# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:39:46 2025

"""
import sys
import os
import numpy as np
import time

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from AspenSim import AspenSim
from CodeLibrary import Simulation

class VCDistillation(AspenSim):
    """
    A class to represent a distillation column in Aspen.
    Inherits from AspenSim and implements the methods for distillation column simulation.
    """
    def __init__(self, AspenFile, wdpath, visibility=False):
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
        for block_type in ["Radfrac"]:
            for block, params in x_dict[block_type].items():
                flat_list.extend(params)
        return np.array(flat_list)

    @staticmethod
    def unflatten_params(flat_array):
        x_dict = {
            "Flash2": {"FLASH1": [flat_array[0]], "FLASH2": [flat_array[1]]},
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

        for blockname, params in x["RadFrac"].items():
            print(params)
            self.sim.BLK_RADFRAC_Set_NSTAGE(blockname, params[0])
            self.sim.BLK_RADFRAC_Set_FeedStage(blockname, params[1][0], params[1][1])
            self.sim.BLK_RADFRAC_Set_Refluxratio(blockname, params[2])
            self.sim.BLK_RADFRAC_Set_DistillateToFeedRatio(blockname, params[3])

        self.sim.DialogSuppression(True)
        self.sim.Run()

        results = {
            "COL_1_DIAM": self.sim.BLK_RADFRAC_Get_Diameter("RADFRAC1"),
            "COL_2_DIAM": self.sim.BLK_RADFRAC_Get_Diameter("RADFRAC2"),
            "COL_1_HEIGHT": self.sim.BLK_RADFRAC_Get_Height("RADFRAC1"),
            "COL_2_HEIGHT": self.sim.BLK_RADFRAC_Get_Height("RADFRAC2"),
            "COL_1_HEAT_UTIL": self.sim.Get_Utility_Cost("LP"),
            "COL_2_HEAT_UTIL": self.sim.Get_Utility_Cost("LP"),
            "COL_1_COOL_UTIL": self.sim.Get_Utility_Cost("REFRIG4"),
            "COL_2_COOL_UTIL": self.sim.Get_Utility_Cost("REFRIG1"),
            "COL_1_REBOILER_DUTY": self.sim.BLK_RADFRAC_Get_ReboilerDuty("RADFRAC1"),
            "COL_2_REBOILER_DUTY": self.sim.BLK_RADFRAC_Get_ReboilerDuty("RADFRAC2"),
        }
        return results
    
    def costFunc(self, results):
        tac = self.calc_tac(results)
        co2_emission = self.calc_co2_emission(results)


        return tac,co2_emission
    
    def calc_tac(self, results):
        # Calculate the total annual cost (TAC) based on the results
        column_1_capital = self.calc_column_cap_cost(results["COL_1_HEIGHT"], results["COL_1_DIAM"])
        column_2_capital = self.calc_column_cap_cost(results["COL_2_HEIGHT"], results["COL_2_DIAM"])
        
        operating_cost_1 = (results["COL_1_HEAT_UTIL"] + results["COL_1_COOL_UTIL"]) * 8000
        operating_cost_2 = (results["COL_2_HEAT_UTIL"] + results["COL_2_COOL_UTIL"]) * 8000
        
        tac = (column_1_capital + column_2_capital)/3+ operating_cost_1 + operating_cost_2
        
        return tac
    
    def calc_column_cap_cost(self, height, diameter):
        return 17640 * diameter**1.066 * height**0.802
    
    def calc_co2_emission(self, results):
        column_1_emission = ((results["COL_1_REBOILER_DUTY"]/0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600
        column_2_emission = ((results["COL_2_REBOILER_DUTY"]/0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600
        total_emission = column_1_emission + column_2_emission
        return total_emission
        
      
    def run_obj(self, x):
        res = self.runSim(x)
        return self.costFunc(res)      
       
import numpy as np
import time

         
def main():
    #! This is the start of the "driver"
    print("Ok we are now testing the VCDistillation class.")
    assSim = VCDistillation(AspenFile="Vinyl Chloride Distillation.bkp", 
                          wdpath="../Vinyl_Distillation", 
                          visibility=False)
    
    #? The order of the paremeters in the dicitonary is as follows:
    #? [number of stages, [feed stage, feed stream name], reflux ratio, distillate to feed ratio]
    # these values are based off the ranges give in the paper
    
    x = {
        'RadFrac': {'RADFRAC1': [32, [15,'FEED'], 1.0, 0.47],
                     'RADFRAC2': [39, [17,'B1'], 1.0, 0.9]},
    }
    
    res = assSim.run_obj(x)
        
    

if __name__ == "__main__":
    main()