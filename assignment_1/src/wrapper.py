# This script serves as a wrapper to run the code for all questions sequentially.
from Q1_Histogram_Computation import main as run_q1
from Q2_Otsus_Binarization import main as run_q2
from Q3_Adaptive_Binarization import main as run_q3
from Q4_Connected_Components import main as run_q4

def run_all():
    
    print("--- Running Question 1: Histogram Computation ---")
    run_q1()
    print("\n--- Question 1 Complete ---\n")

    print("--- Running Question 2: Otsu's Binarization ---")
    run_q2()
    print("\n--- Question 2 Complete ---\n")

    print("--- Running Question 3: Adaptive Binarization ---")
    run_q3()
    print("\n--- Question 3 Complete ---\n")

    print("--- Running Question 4: Connected Components ---")
    run_q4()
    print("\n--- Question 4 Complete ---\n")

    print("All processes finished successfully!")

if __name__ == '__main__':
    run_all()