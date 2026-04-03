import json

def process_train():
    processed_train = []
    with open("/home/bkzhu/storage/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt = f.read()

    with open("/home/bkzhu/storage/assignment5-alignment/data/gsm8k/train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            train_sample = {}
            line = json.loads(line)
            train_sample["prompt"] = prompt.format(question=line["question"])
            train_sample["response"] = "<think> " + line["answer"].replace("\n####", " </think> <answer>") + " </answer>"
            processed_train.append(train_sample)

    with open("/home/bkzhu/storage/assignment5-alignment/data/gsm8k/processed_train.jsonl", "w", encoding="utf-8") as f:
        for item in processed_train:
            f.write(json.dumps(item) + "\n")


def make_correct_data():
    correct = []
    format_correct_answer_wrong = []
    format_wrong_answer_correct = []
    format_wrong_answer_wrong = []
    with open("baseline_result.jsonl") as f:
        for line in f:
            line = json.loads(line)
            if line["format_reward"] == 1 and line["answer_reward"] == 1:
                correct.append(line)
            elif line["format_reward"] == 0 and line["answer_reward"] == 1:
                format_wrong_answer_correct.append(line)
            elif line["format_reward"] == 1 and line["answer_reward"] == 0:
                format_correct_answer_wrong.append(line)
            else:
                format_wrong_answer_wrong.append(line)
                format_wrong_answer_wrong.append(line)
                format_wrong_answer_wrong.append(line)
    print(len(correct))
    print(len(format_correct_answer_wrong))
    print(len(format_wrong_answer_correct))
    print(len(format_wrong_answer_wrong))

    correct_dict_list = []
    for i in correct:
        correct_dict = {}
        correct_dict["prompt"] = i["prompt"]
        correct_dict["response"] = i["response"]
        correct_dict_list.append(correct_dict)

    with open("/home/bkzhu/storage/assignment5-alignment/data/gsm8k/train_correct.jsonl", "w") as f:
        for i in correct_dict_list:
            json.dump(i, f)
            f.write("\n")
if __name__ == "__main__":
    # process_train()
    make_correct_data()