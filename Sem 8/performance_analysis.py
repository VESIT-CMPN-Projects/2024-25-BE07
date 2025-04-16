# performance_analysis.py

from my_rag_for_stationcodes_entire_human_dynamic_withTrainStatus import retrieve_documents, run_rag_pipeline
import matplotlib.pyplot as plt

def sensitivity_analysis(query):
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    accuracies = []
    times = []
    
    for threshold in thresholds:
        print(f"\nRunning for threshold {threshold}")
        response, avg_distance, query_time = run_rag_pipeline(query, threshold=threshold, max_k=10)
        accuracies.append(avg_distance)  # Using average distance as a proxy for accuracy
        times.append(query_time)

    # Plotting Accuracy vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(times, accuracies, marker='o', color='b')
    plt.title('Accuracy (Average Distance) vs Time Taken for Query')
    plt.xlabel('Time Taken (seconds)')
    plt.ylabel('Accuracy (lower is better, avg distance)')
    plt.grid(True)
    plt.show()

# Example of running the sensitivity analysis
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    sensitivity_analysis(user_query)
