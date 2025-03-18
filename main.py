from task_extractor import TaskExtractor

def main():
    # Sample text
    sample_text = """
    Rahul and Vaibhav must submit the final assignment by next Monday and presentation by Friday.
    Satyam needs to review the report by April 15th.
    If approved, Jiya should schedule a team meeting for next week.
    Makram is responsible for cleaning the kitchen this weekend.
    Please submit the monthly expense reports by the end of this month.
    """

    # Initialize task extractor
    extractor = TaskExtractor()

    # Extract tasks
    tasks = extractor.extract_tasks(sample_text)

    # Print results
    print(f"Extracted {len(tasks)} tasks:")
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}:")
        print(f"  Description: {task['task_description']}")
        print(f"  Person: {', '.join(task['person'])}")
        print(f"  Deadline: {', '.join(task['deadline'])}")
        print(f"  Category: {task.get('category', 'Uncategorized')}")
        if "topic_distribution" in task:
            print("  Topic distribution:")
            for topic, prob in task["topic_distribution"].items():
                print(f"    - {topic}: {prob}")

if __name__ == "__main__":
    main()