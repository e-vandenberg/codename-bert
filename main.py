import torch
from guesser import SiameseBert

if __name__ == '__main__':
    model = SiameseBert()
    state_dict = torch.load("saved_model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    input_word = "butterfly"
    board = ["radiator", "meteor", "truck",
            "runner", "flower", "float",
            "goat", "theatre", "guitar"]

    outputs = {}
    for word in board:
        # evaluate input word against board words
        output = model(input_word, word).item()
        outputs[word] = output

    print(outputs)
