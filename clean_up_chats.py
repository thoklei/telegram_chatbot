from bs4 import BeautifulSoup
import numpy as np
import os
import argparse 

def main(path_to_files, num_files):
    filenames = ["messages"]+["messages"+str(i) for i in range(2, num_files+1)] #["messages", "messages2", ...]
    target = []
    f = open(os.path.join(path_to_files, "result.txt"), 'w')

    for filename in filenames:
        soup = BeautifulSoup(open(os.path.join(path_to_files, filename+".html")), 'html.parser')

        divs = soup.find_all('div')
        author = ""

        for d in divs:
            if(d['class'] == ['from_name']):
                author = str(d.contents[0]).strip().split()[0]+": "
            elif(d['class'] == ['text']):
                for sentence in d.contents:
                    sentence = str(sentence)
                    if(not sentence == "<br/>" and not sentence.isspace()):
                        if(sentence.startswith("<a")):
                            sentence = "link"
                        f.write(author+' '.join(sentence.strip().split()[0:100]).replace("&apos","'")+"\n")
            

    f.close()

    def remove_authors(path_to_files):
        with open(os.path.join(path_to_files,"result.txt")) as f:
            content = f.readlines()
        clean = [line.split(": ")[1] for line in content]
        with open(os.path.join(path_to_files,"result_clean.txt"),"w") as f:
            f.writelines(clean)

    remove_authors(path_to_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates training dataset from Telegram chats.')
    parser.add_argument("-p", type=str,
                    help='The path to the message.html files that should be processed')
    parser.add_argument('-n', type=int,
                    help='How many files are there?')

    args = parser.parse_args()
    main(args.p, args.n)
