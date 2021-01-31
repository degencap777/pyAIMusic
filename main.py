import time
import tensorflow as tf
import music21 as m21
import numpy as np
# import cupy as cp #TODO: install cupy?
import os
from collections import Counter
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import keras.layers as klayers
import keras.models as kmodels
import keras.callbacks as kcallbacks
import keras.backend as kbackend
import random
from progress.bar import ShadyBar, Bar, PixelBar, ChargingBar
from progress.spinner import Spinner
from multiprocessing import Process

from models import lstm
from utils import longseq_ingest, batch_ingest

'''
TODO:
1 - use cupy
2 - split into multiple pytho
'''


def check_inputs():
    """
    this will return filepaths, ingestions, etc.
    :return: ingest, data_path, re_fit, weights_name, output_name
    """
    ingest = True  # keep this true by default just in case.
    data_path = "music\\"
    re_fit = True  # keep this true by default. Determines whether or not to re-train model.
    weights_name = "best_model.h5"
    output_name = "prediction"
    while (True):
        ingest_check = input("do you need to ingest midi files? (y/n)")
        if ingest_check == 'y' or ingest_check == 'Y':
            ingest = True
            break
        elif ingest_check == 'n' or ingest_check == 'N':
            ingest = False
            break
        else:
            print("you entered an invalid character.")
            continue
    if ingest:
        while (True):
            inp_check = input("the current dataset path is: " + data_path + ". Would you like to change path? (y/n)")
            if inp_check == 'y' or inp_check == 'Y':
                data_path = input("please enter your full path here:")
            elif inp_check == 'n' or inp_check == 'N':
                print("ok, using default path")
                break
            else:
                print("you have entered an invalid character.")
    while (True):
        inp_check = input("do you want to re-fit your mode? (y/n)")
        if inp_check == 'y' or inp_check == 'Y':
            re_fit = True
            break
        elif inp_check == 'n' or inp_check == 'N':
            re_fit = False
            break
        else:
            print("you have entered an invalid character.")

    while (True):
        inp_check = input("the current model_name is: " + weights_name + ".h5 Would you like to change name? (y/n)")
        if inp_check == 'y' or inp_check == 'Y':
            weights_name = input("please enter your desired name here (without extension):")
        elif inp_check == 'n' or inp_check == 'N':
            print("ok, using current model_name " + weights_name + ".h5")
            break
        else:
            print("you have entered an invalid character.")

    while (True):
        inp_check = input("the current output name is: " + output_name + ". Would you like to change name? (y/n)")
        if inp_check == 'y' or inp_check == 'Y':
            output_name = input("please enter your desired name here (without extension):")
        elif inp_check == 'n' or inp_check == 'N':
            print("ok, using current model_name " + output_name + ".")
            break
        else:
            print("you have entered an invalid character.")
    return ingest, data_path, re_fit, weights_name, output_name


def fit_wavenet(x_tr, x_val, y_tr, y_val):
    # xx% for training and 100-xx% for evaluation.
    '''currently using 20% for evaluation'''
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
    # TF_XLA_FLAGS = --tf_xla_auto_jit = 1 #enable xla
    # x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)
    print("building model")
    # start building the model now. Clean this up later.
    kbackend.clear_session()
    model = kmodels.Sequential();

    # embedding layers
    model.add(klayers.Embedding(len(unique_x), 100, input_length=32, trainable=True))

    model.add(klayers.Conv1D(64, 3, padding='causal', activation='relu'))
    model.add(klayers.Dropout(0.2))
    model.add(klayers.MaxPool1D(2))

    model.add(klayers.Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
    model.add(klayers.Dropout(0.2))
    model.add(klayers.MaxPool1D(2))

    model.add(klayers.Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
    model.add(klayers.Dropout(0.2))
    model.add(klayers.MaxPool1D(2))

    # model.add(Conv1D(256,5,activation='relu'))
    model.add(klayers.GlobalMaxPool1D())

    model.add(klayers.Dense(256, activation='relu'))
    model.add(klayers.Dense(len(unique_y), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()

    # callback to save best model during training
    # TODO: remember to change the names
    mc = kcallbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
    model.fit(np.array(x_tr), np.array(y_tr), batch_size=64, epochs=20,
              validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])
    # model = kmodels.load_model('best_model.h5')

def fit_lstm(splitter):
    x_tr, x_val, y_tr, y_val = splitter
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
    # TF_XLA_FLAGS = --tf_xla_auto_jit = 1  # enable xla
    kbackend.clear_session()
    model = kmodels.Sequential()
    model.add(klayers.LSTM(128, return_sequences=True))
    model.add(klayers.LSTM(128))
    model.add(klayers.Dense(256))
    model.add(klayers.Activation('relu'))
    model.add(klayers.Dense(128))
    model.add(klayers.Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    # model.summary()
    mc = kcallbacks.ModelCheckpoint('best_lstm.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    model.fit((np.array(x_tr), np.array(y_tr)), batch_size=64, epochs=20,
              validation_data=(np.array(x_val).any(), np.array(y_val).any()), verbose=1, callbacks=[mc])
    model.summary()

def fit_model(type, splitter):
    if type == "lstm":
        fit_lstm(splitter)
    elif type == "wavenet":
        x_tr, x_val, y_tr, y_val = splitter
        fit_wavenet(x_tr, x_val, y_tr, y_val)


def convert_to_midi(prediction_output, output_path):
    """
    converst back to midi !
    :param prediction_output:
    :return:
    """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                cn = int(current_note)
                new_note = m21.note.Note(cn)
                new_note.storedInstrument = m21.instrument.Piano()
                notes.append(new_note)

            new_chord = m21.chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # pattern is a note
        else:

            new_note = m21.note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = m21.instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = m21.stream.Stream(output_notes)
    output_path = output_path + '.mid'
    midi_stream.write('midi', fp=output_path)

def start_spinner():
    print("start spinner \n \n")
    spinner = Spinner("loading")
    while spinner_bool:
        # time.sleep(1)
        spinner.next()

# Press the green button in the gutter to run the script.
spinner_bool = False
if __name__ == '__main__':

    # # read_midi('F:\\Winter2021\\PythonMusicAI\\dataset\\schu_143_1.mid')
    path = 'music\\' #local referenced path :!
    num_timesteps = 32  # you need to change input length if you change this too
    # # path = 'dataset/'

    ingest = False  # do you want to ingest?
    re_fit = True  # do you want to re-fit the model?
    graph_frequency = False  # graph frequency of notes?
    load_splitter = True;
    output = 'predicted_tuna_l2'  # what would you like your output name to be?
    prediction_len = 500  # how many steps of prediction do you want

    # ingest, path, re_fit, weights_name, output_name = check_inputs()
    # ingest = true if want ingest, false if no need to ingest
    # path = dataset path
    # re_fit = true if model needs to be fitted again, false if already fit.
    # weights_name = checkpoint model name
    # output_name = output of predictions.

    if ingest:
        # batch_ingest(dir=path)
        longseq_ingest(path=path, num_timesteps=num_timesteps,thresh=50, graph=graph_frequency)
        # ingest = Process(target=ingest_to_csv, args=(path, num_timesteps, graph_frequency))
        # spinner = Process(target=start_spinner, args=())
        #
        # ingest.start()
        #
        # ingest.join()
        # spinner.join()
        # #
        # # spinner.start()
        # # spinner.join()
        print("ingested")

        # while(~spinner_bool):
        #     ingest.terminate()
        #     spinner.terminate()
        #     break

    # print("memes1")
    # files = [i for i in os.listdir(path) if i.endswith(".mid")]
    # print("memes2")
    # print(len(files))
    # notes_array = np.array([read_midi(path + i) for i in files], dtype=object)
    # print("memes3")
    # notes_ = [element for note_ in notes_array for element in note_]
    # print("memes4")
    #
    # unique_notes = list(set(notes_))
    # print(len(unique_notes))
    #
    # freq = dict(Counter(notes_))
    #
    # num = [count for _, count in freq.items()]
    #
    # plot.figure(figsize=(5, 5))
    #
    # print("plotting")
    # plot.hist(num)
    # plot.show()
    # print("plot done")
    #
    # frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
    # print(len(frequent_notes))
    #
    # # prepare musical files with only top freq notes
    #
    # new_music = []
    # for notes in notes_array:
    #     temp = []
    #     for note_ in notes:
    #         if note_ in frequent_notes:
    #             temp.append(note_)
    #         new_music.append(temp)
    # new_music = np.array(new_music, dtype=object)
    # print("new_music done")
    #
    # # prepping data
    # num_timesteps = 32
    # x = []
    # y = []
    #
    # print(len(new_music))
    # print(len(note_))
    # for note_ in new_music:
    #     for i in range(0, len(note_) - num_timesteps, 1):
    #         # print("note_ i:")
    #         # print("note_ i: ", i)
    #         # prep input/outpu seqs
    #         input_ = note_[i:i + num_timesteps]
    #         output = note_[i + num_timesteps]
    #
    #         x.append(input_)
    #         y.append(output)
    # print("arrays appended")
    # x = np.array(x)
    # y = np.array(y)  # bc np arrays >>>>>
    #
    # print("x/y arrays made")
    # # asign intger to every note
    # unique_x = list(set(x.ravel()))
    # x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
    #
    # # prepare integer sequences for input data
    # x_seq = []
    # for i in x:
    #     temp = []
    #     print("i:")
    #     print(i)
    #     for j in i:
    #         # assign int to note
    #         temp.append(x_note_to_int[j])
    #         print("j:")
    #         print(j)
    #     x_seq.append(temp)
    # x_seq = np.array(x_seq)
    # print("x seq done")
    #
    # # prepareinteger sequences for output data
    # unique_y = list(set(y))
    # y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    # y_seq = np.array([y_note_to_int[i] for i in y])
    # print("prepped")

    # this block is still needed for predictions later :(
    print("loading arrays")
    x_seq = np.load("x_seq.npy")
    y_seq = np.load("y_seq.npy")
    unique_x = np.load("x_unique.npy")
    unique_y = np.load("y_unique.npy")
    print("loaded")

    if load_splitter:
        x_tr = np.load("x_tr.npy")
        x_val = np.load("x_val.npy")
        y_tr = np.load("y_tr.npy")
        y_val = np.load("y_val.npy")
    else:
        # xx% for training and 100-xx% for evaluation.
        '''currently using 20% for evaluation'''
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
        x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)
        np.save("x_tr", x_tr)
        np.save("x_val", x_val)
        np.save("y_tr", y_val)
        np.save("y_val", y_val)
    splitter = x_tr, x_val, y_tr, y_val
    print("done train test split")

    '''
    ok so:
    lstm model callback: best_lstm.h5
    wavenet model callback: best_model.h5
    '''
    if re_fit:
        fit_model("lstm", splitter)
    model = kmodels.load_model('best_lstm.h5')

    # now we compose our own music......
    ind = random.randint(0, len(x_val) - 1)

    # ind2 = np.random.randint(0, len(x_val)-1)

    random_music = x_val[ind]

    # random_music2 = x_val[ind2]

    predictions = []
    for i in range(prediction_len):
        random_music = random_music.reshape(1, num_timesteps)

        prob = model.predict(random_music)[0]
        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
        random_music = random_music[1:]
    print(predictions)

    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
    predicted_notes = [x_int_to_note[i] for i in predictions]

    # everything should be in a main function.
    convert_to_midi(predicted_notes, output)
# FluidSynth.midi_to_audio('predicted.mid', 'output.wav')  # hopefully this generates a wav file.
