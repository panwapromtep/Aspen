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
        for block_type in ["RadFrac"]:
            for block, params in x_dict[block_type].items():
                flat_list.extend(params)
        return np.array(flat_list)

    @staticmethod
    def unflatten_params(flat_array):
        print('hi')
        radfrac1 = flat_array[:4]
        radfrac2 = flat_array[-4:]
        x_dict = {
            "RadFrac": {"RADFRAC1": [radfrac1[0], [radfrac1[1], 'FEED'], radfrac1[2], radfrac1[3]], 
                        "RADFRAC2": [radfrac2[0], [radfrac2[1], 'B1'], radfrac2[2], radfrac2[3]]}
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
            end_stage_2 = params[0] -1

            
            self.sim.BLK_RADFRAC_Set_NSTAGE(blockname, params[0])
            self.set_section_value(blockname, end_stage_2)
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
            "ACETYLENE_PURITY": self.sim.get_acetylene_purity("B1"),
            "VC_PURITY": self.sim.get_vc_purity("D2"),
        }
        print(results.items())
        return results
    
    def costFunc(self, results):
        tac = self.calc_tac(results)
        co2_emission = self.calc_co2_emission(results)        
        return (results["ACETYLENE_PURITY"], results["VC_PURITY"], tac, co2_emission)
    
    def calc_tac(self, results):
        # Calculate the total annual cost (TAC) based on the results
        column_1_capital = self.calc_column_cap_cost(results["COL_1_HEIGHT"], results["COL_1_DIAM"])
        column_2_capital = self.calc_column_cap_cost(results["COL_2_HEIGHT"], results["COL_2_DIAM"])
        column_1_heatex_capital = self.sim.get_heat_exchanger_cost("RADFRAC1")
        column_2_heatex_capital = self.sim.get_heat_exchanger_cost("RADFRAC2")
        
        operating_cost_1 = (results["COL_1_HEAT_UTIL"] + results["COL_1_COOL_UTIL"]) * 8000
        operating_cost_2 = (results["COL_2_HEAT_UTIL"] + results["COL_2_COOL_UTIL"]) * 8000
        
        tac = (column_1_capital + column_2_capital + column_1_heatex_capital + column_2_heatex_capital)/3+ operating_cost_1 + operating_cost_2
        return tac
    
    def calc_column_cap_cost(self, height, diameter):
        return 17640 * diameter**1.066 * height**0.802
    
    def calc_co2_emission(self, results):
        column_1_emission = ((results["COL_1_REBOILER_DUTY"]/0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600 * 0.00110231 #converts kilograms to tons
        column_2_emission = ((results["COL_2_REBOILER_DUTY"]/0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600 * 0.00110231
        total_emission = column_1_emission + column_2_emission
        return total_emission

    def set_section_value(self, blockname: str, new_value):
        """
        Set the INT‑1 → CS‑2 section value under Column Internals for the given block.
        """
        path = (
            f"\\Data\\Blocks\\{blockname}"
            "\\Subobjects\\Column Internals\\INT-1"
            "\\Subobjects\\Sections\\CS-2"
            "\\Input\\CA_STAGE2\\INT-1\\CS-2"
        )
        node = self.sim.AspenSimulation.Tree.FindNode(path)
        node.Value = new_value

        
      
    def run_obj(self, x):
        res = self.runSim(x)
        return self.costFunc(res)     
    
from pymoo.core.problem import Problem
import torch
class VinylDistillationNNProblem(Problem):
    def __init__(self, model, scaler):
        # n_var = 8 input variables, n_obj = 2 objectives.
        # Set vectorized=True so that _evaluate receives a matrix of solutions.
        super().__init__(n_var=8, n_obj=2, n_ieq_constr = 2,xl=[-1]*8, xu=[1]*8, vectorized=True)
        self.model = model
        thresholds = scaler.transform(torch.tensor([0.0005, 0.9999], dtype=torch.float32)).numpy()
        acetyl_threshold, vc_threshold = thresholds[0], thresholds[1]
        print("acetyl_threshold", acetyl_threshold)
        print("vc_threshold", vc_threshold)
        self.acetyl_threshold = acetyl_threshold
        self.vc_threshold = vc_threshold

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Vectorized evaluation of candidate solutions.

        Parameters:
            X: A 2D NumPy array of shape (n, 8), where each row is a candidate solution.
            out: A dictionary where results (objectives) should be stored under key "F".
        """
        # Convert X into a torch tensor. X is assumed to be a numpy array with shape (n, 8)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            # Evaluate the model on the entire batch.
            # The model should return a tensor of shape (n, 2)
            output = self.model(X_tensor).numpy()
            
        #first column is acetylene purity, second column is vinyl chloride purity, third column is the cost,   4th column is the co2 emission

        g1 = output[:, 0] - self.acetyl_threshold  # Acetylene purity constraint
        g2 = self.vc_threshold - output[:, 1]  # Vinyl chloride purity constraint
        
        out["G"] = np.column_stack((g1, g2))  # Store constraints in the output dictionary
        
        # Store the objectives in the output dictionary
        out["F"] = np.column_stack((output[:, 2], output[:, 3]))
      
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