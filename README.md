# telegram_chatbot
A telegram chatbot, trained on your own chat history.

So, there is this chat app that I really like called Telegram, which is basically Whatsapp with a better UI. One of its coolest features is that it allows you to program bots: Little programs that use Telegram's API to appear like other users that you can send messages to and receive answers from. The most obvious project you could use that for is a chatbot that was trained on the real messages you exchanged with friends. So let's give it a try.

# Exporting your chat data
You can do that by entering a chat, clicking on the menu and choosing "export chat history". You can also export images, voice messages and other stuff, but we only need the raw messages. Telegram will create a folder with a bunch of html files and some css, which allows for a rather pretty depiction of the messages, but we don't need that, we only want the raw text. So I wrote a script to filter the text out, using beautiful soup, which you might have to install. You can start the script from the command line, just tell it where to look and how many "messages_n.html" files there are. It will produce two outputs: One in which the author is part of the message, and one with only the messages. 

# Setting up the bot
You will need an API key (or "access token") to register your bot. You can use Telegram's Botfather to guide you through the extremely simple setup process. 

# A simple echo-bot
Now we need a script that interacts with the API. There are dozens of ways of doing this, several for every supported language. The example that I provide here is the simplest one I could find: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/echobot.py

It basically just consists of an echo-function that responds to every message with the text of that message. So if you text "Hi!", you're gonna get "Hi!" back. The first change I made to it is reading in the access token from a textfile instead of putting it into the code directly - I did that to keep my access token private while still being able to keep the code on github. Exchange that for your access token in line 45. 

# Adding a language model
The heart of the bot is of course the language model. All I'm doing in there is taken from the Tensorflow tutorial for text generation, you can find that here: https://www.tensorflow.org/tutorials/sequences/text_generation. The only difference is that I load the model from a .h5 file, which contains the full model, the architecture as well as the weights (and the optimizer state, for what it's worth, but we don't need that here). 

The reason for why I did it that way should be obvious: We can easily exchange the model by changing the file path from which the model is instantiated. You could even add a command to change the model that is used - if you have different models that were trained on different friends of yours, that would be like changing the "personality" of the bot. Just keep in mind that you'd have to change the textfile as well (if I find the time, I'll change it so that we use a pickled python dictionary or something like that for the vocabulary of the model, but for now, this should work.)

# Training the language model
Of course, the hardest part is building and training the model. In this first version, I used a simple (although fairly large, if you are training on CPU) LSTM with one layer and 1024 recurrent units. This step is simplified considerably by the fact that there is an excellent tutorial that does all the work for us - I just removed the explanatory bits and put everything into a .py-file, made sure that models are stored in a way that allows us to completely restore them from the checkpoint and added a custom callback to store a second version of the model after every epoch. This version has a batchsize of 1, so it can be used by the bot. 

# Command line arguments
This file can be started from the command line with a couple of arguments to specify what exactly we want to do:
 - scratch: set it to 1 if you want to train from scratch, set it to 0 if you want to start from a checkpoint
 - checkpoint: the exact checkpoint file you want to start from (e.g. "checkpoint.h5"
 - checkpointdir: the directory in wich the generated checkpoints should be stored
 - epochs: for how many epochs you want to train the model
 - textfile: the file to use for creating the dictionary (this needs to be the file you trained on)
 
 # Finally, here are a bunch of things I learned while doing this:
 - if you are using sparsecrossentropy, make sure to specify whether your inputs are logits or not, default is False and it makes a huge difference (as in, doesn't work if wrong)
 - argparse cannot handle boolean input. You can specify type=bool but the result will always be True, no matter what you put in
 - when doing text generation, *sample from the distribution instead of taking the argmax*. Taking the argmax results in disappointing nonsensical loops (been there, done that, super frustrating)
