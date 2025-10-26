# This script serves as a wrapper to run the code for all questions sequentially.
from Q1_Directional_Filtering import main as run_q1
from Q2_Gaussian_Blurring import main as run_q2

def run_all():
    
    print("--- Running Question 1: Directional Filtering ---")
    run_q1()
    print("\n--- Question 1 Complete ---\n")

    print("--- Running Question 2: Gaussian Blurring ---")
    run_q2()
    print("\n--- Question 2 Complete ---\n")

    print("All processes finished successfully!")

if __name__ == '__main__':
    run_all()