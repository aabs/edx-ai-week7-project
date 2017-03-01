import sys


def open_input(input_file):
    fo = open(input_file, "r")
    return fo


def open_output(output_file):
    fo = open(output_file, "w")
    return fo


class Perceptron:
    def __init__(self, fo_in, fo_out):
        pass

    def run(self):
        pass


def main():
    input = open_input(sys.argv[1])
    output = open_output(sys.argv[2])
    pla = Perceptron(input, output)
    pla.run()
    return 0
