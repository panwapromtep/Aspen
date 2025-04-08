import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
sys.path.append(parent_dir)

from AspenSim import AspenSim
from CodeLibrary import Simulation

def main():
    #! This is the start of the "driver"

    print("Ok we are now testing setters and getts for the Vinyl Distillation Column simulation.")
    
    sim = Simulation(AspenFileName="Vinyl Chloride Distillation.bkp", WorkingDirectoryPath="../Vinyl_Distillation", VISIBILITY=False)
    
    diameter = sim.BLK_RADFRAC_Get_Diameter('RADFRAC1')
    print(f"Current diameter: {diameter}")
    
    height = sim.BLK_RADFRAC_Get_Height('RADFRAC1')
    print(f"Current height: {height}")
    
    refrigeration_cost = sim.Get_Utility_Cost('REFRIG4')
    print(f"Current refrigeration cost: {refrigeration_cost}")
    
    reboiler_duty = sim.BLK_RADFRAC_Get_ReboilerDuty('RADFRAC1')
    print(f"Current reboiler duty: {reboiler_duty}")
    
    
    #testing getters and setters
    num_stage = sim.BLK_RADFRAC_Get_NSTAGE("RADFRAC1")
    print(f"Current Number of Stages: {sim.BLK_RADFRAC_Get_NSTAGE('RADFRAC1')}")

    #Example of getting feed position
    feed_stage = sim.BLK_RADFRAC_Get_FeedStage("RADFRAC1")
    print(f"Current feed stage: {feed_stage}")


    # Example of getting the reflux ratio
    reflux_ratio = sim.BLK_RADFRAC_Get_Refluxratio("RADFRAC1")
    print(f"Current reflux ratio: {reflux_ratio}")
    

    #example of getting distillate to feed ratio
    distillate_to_feed_ratio = sim.BLK_RADFRAC_Get_DistillateToFeedRatio("RADFRAC1")
    print(f"Current distillate to feed ratio: {distillate_to_feed_ratio}")

    
    #testing the update to them   
    sim.BLK_RADFRAC_Set_NSTAGE("RADFRAC1", num_stage + 1) # Example of setting the number of stages
    print(f"Number of stages after increment: {sim.BLK_RADFRAC_Get_NSTAGE('RADFRAC1')}")
    sim.BLK_RADFRAC_Set_Refluxratio("RADFRAC1", reflux_ratio + 0.1)  # Example of setting a new reflux ratio
    print(f"New reflux ratio set to: {sim.BLK_RADFRAC_Get_Refluxratio('RADFRAC1')}")
    sim.BLK_RADFRAC_Set_FeedStage("RADFRAC1", feed_stage + 1, "FEED")  # Example of setting a new feed stage
    print(f"New feed stage set to: {sim.BLK_RADFRAC_Get_FeedStage('RADFRAC1')}")
    sim.BLK_RADFRAC_Set_DistillateToFeedRatio("RADFRAC1", distillate_to_feed_ratio + 0.05)  # Example of setting a new ratio
    print(f"New distillate to feed ratio set to: {sim.BLK_RADFRAC_Get_DistillateToFeedRatio('RADFRAC1')}")
    
    
    sim.SaveAs("updated.apwz", True)  # Save the updated Aspen file
    print("The Aspen file has been saved as 'updated.apwz'.")
    
    # calculating the economic objective function
    sim.CloseAspen() # Close the simulation to free up resources
    
    

if __name__ == "__main__":
    main()