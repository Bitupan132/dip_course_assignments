# This script serves as a wrapper to run the code for all questions sequentially.
from Q1_Spatial_Filtering_Binarization import main as run_q1
from Q2_Scaling_Rotation_Interpolation import main as run_q2
from Q3_Sharpen import main as run_q3

def run_all():
    
    print("--- Running Question 1: Spatial Filtering and Binarization ---")
    run_q1()
    print("\n--- Question 1 Complete ---\n")

    print("--- Running Question 2: Scaling and Rotation with Interpolation ---")
    run_q2()
    print("\n--- Question 2 Complete ---\n")

    print("--- Running Question 3: Image Shareping ---")
    run_q3()
    print("\n--- Question 3 Complete ---\n")

    print("All processes finished successfully!")

if __name__ == '__main__':
    run_all()