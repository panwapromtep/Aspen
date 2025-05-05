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
        radfrac1 = flat_array[:4]
        radfrac2 = flat_array[-4:]
        x_dict = {
            "RadFrac": {"RADFRAC1": [radfrac1[0], [radfrac1[1], 'FEED'], radfrac1[2], radfrac1[3]], 
                        "RADFRAC2": [radfrac2[0], [radfrac2[1], 'B1'], radfrac2[2], radfrac2[3]]}
        }
        return x_dict
    
    def open_simulation(self):
        if self.sim is None:
            # build full path to the Aspen archive
            archive = os.path.join(self.wdpath, self.AspenFile)
            if not os.path.isfile(archive):
                raise FileNotFoundError(f"Aspen archive not found at {archive!r}")
            # now launch Aspen with the correct filename and working dir
            self.sim = Simulation(
                AspenFileName=archive,
                WorkingDirectoryPath=self.wdpath,
                VISIBILITY=self.visibility
            )


    def close_simulation(self):
        if self.sim:
            self.sim.CloseAspen()
            self.sim = None
            

     
    def runSim(self, x):
        try:
            self.open_simulation()
        
            for blockname, params in x["RadFrac"].items():
                print(blockname, params)
                # integer stage values
                nstage    = round(params[0])
                feed_pos  = round(params[1][0])
                feed_port = params[1][1]
        
                # round the continuous knobs to 4 decimals
                reflux   = round(float(params[2]), 4)
                dist2feed = round(float(params[3]), 4)

                # -- commpute cs1 bound
                cs1_start = 2
                cs1_end = feed_pos - 1 if feed_pos > cs1_start else cs1_start

                # --compute cs2 bound
                cs2_start = cs1_end + 1
                cs2_end = nstage - 1

        
                # apply to Aspen
                self.sim.BLK_RADFRAC_Set_NSTAGE(blockname, nstage)
                self.set_sectioncs1_value(blockname, cs1_end)
                
                self.set_sectioncs2_value(blockname, cs1_start, cs2_end)
        
                # the rest of your block settings…
                self.sim.BLK_RADFRAC_Set_FeedStage(blockname, feed_pos, feed_port)
                self.sim.BLK_RADFRAC_Set_Refluxratio(blockname, reflux)
                self.sim.BLK_RADFRAC_Set_DistillateToFeedRatio(blockname, dist2feed)
        
            self.sim.DialogSuppression(True)
            time.sleep(2)
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
            
            self.close_simulation()
            return results
        except Exception as e:
            print(f"Error during Aspen simulation: {e}")
            # Return penalty values for objectives and constraints
            return {
                "COL_1_DIAM": 1e6,  # Large penalty values
                "COL_2_DIAM": 1e6,
                "COL_1_HEIGHT": 1e6,
                "COL_2_HEIGHT": 1e6,
                "COL_1_HEAT_UTIL": 1e6,
                "COL_2_HEAT_UTIL": 1e6,
                "COL_1_COOL_UTIL": 1e6,
                "COL_2_COOL_UTIL": 1e6,
                "COL_1_REBOILER_DUTY": 1e6,
                "COL_2_REBOILER_DUTY": 1e6,
                "ACETYLENE_PURITY": 0.5,  # Invalid purity values
                "VC_PURITY": 0.5,
            }
    
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
        column_1_emission = ((results["COL_1_REBOILER_DUTY"]/0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600 
        column_2_emission = ((results["COL_2_REBOILER_DUTY"]/0.8) / 22000) * 0.0068 * 3.67 * 8000 * 3600
        total_emission = column_1_emission + column_2_emission
        return total_emission
        
    def set_sectioncs1_value(self, blockname: str, new_value):
        """
        Set the INT‑1 → CS‑1 section value under Column Internals for the given block.
        """
        path = (
            f"\\Data\\Blocks\\{blockname}"
            "\\Subobjects\\Column Internals\\INT-1"
            "\\Subobjects\\Sections\\CS-1"
            "\\Input\\CA_STAGE2\\INT-1\\CS-1"
        )
        node = self.sim.AspenSimulation.Tree.FindNode(path)
        node.Value = new_value
        
    def set_sectioncs2_value(self, blockname: str, new_start, new_end):
        """
        Set the INT‑1 → CS‑2 ‘start stage’ value (CA_STAGE1) and CS-2 'end stage' value (CA_STAGE2) under Column Internals for the given block.
        This method also checks to make sure start stage is less than end stage and protects against invalid values.
        """
        start_path = (
            f"\\Data\\Blocks\\{blockname}"
            "\\Subobjects\\Column Internals\\INT-1"
            "\\Subobjects\\Sections\\CS-2"
            "\\Input\\CA_STAGE1\\INT-1\\CS-2"
        )
        
        end_path = (
            f"\\Data\\Blocks\\{blockname}"
            "\\Subobjects\\Column Internals\\INT-1"
            "\\Subobjects\\Sections\\CS-2"
            "\\Input\\CA_STAGE2\\INT-1\\CS-2"
        )
        start_node = self.sim.AspenSimulation.Tree.FindNode(start_path)
        end_node = self.sim.AspenSimulation.Tree.FindNode(end_path)
        
        if (new_end < start_node.Value): #if the new end is less than the prev start, then change the start first
            start_node.Value = new_start
            end_node.Value = new_end
        else:
            end_node.Value = new_end
            start_node.Value = new_start


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
    
    #? The order of the paremeters in the dictionary is as follows:
    #? [number of stages, [feed stage, feed stream name], reflux ratio, distillate to feed ratio]
    # these values are based off the ranges give in the paper
    
    x = {
        'RadFrac': {'RADFRAC1': [32.85437159720812, [32.51295556835407, 'FEED'], 1.14998209356873, 0.4489752914540106],
                     'RADFRAC2': [44.77891153287981, [34.44278350177228,'B1'], 1.3823604710569735, 0.7064783109609141]},
    }

    
    a = {
        'RadFrac': {'RADFRAC1': [39, [19,'FEED'], 0.7140, 0.4620],
                     'RADFRAC2': [41, [35,'B1'], 0.3320, 0.8930]},
    }
    
    b = {
        'RadFrac': {'RADFRAC1': [39, [28,'FEED'], 0.8760, 0.4700],
                     'RADFRAC2': [42, [25,'B1'], 0.2870, 0.8960]},
    }
    
    c = {
        'RadFrac': {'RADFRAC1': [40, [23,'FEED'], 0.5540, 0.4718],
                     'RADFRAC2': [41, [9,'B1'], 0.3903, 0.8930]},
    }
    
    d = {
        'RadFrac': {'RADFRAC1': [38, [23,'FEED'], 0.6915, 0.4623],
                     'RADFRAC2': [40, [27,'B1'], 0.3248, 0.9023]},
    }
    
    res = assSim.run_obj(d)
    
    print("TAC:", res[2])
    print("CO2 Emission:", res[3])
    
        
    

if __name__ == "__main__":
    main()