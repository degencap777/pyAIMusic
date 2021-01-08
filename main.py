import music21 as m21
import numpy as np
import cupy as cp
import os
from collections import Counter
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import keras.layers as klayers
import keras.models as kmodels
import keras.callbacks as kcallbacks
import keras.backend as kbackend
import random
from midi2audio import FluidSynth
import tensorflow as tf


def check_inputs():
    """
    this will return filepaths, ingestions, etc.
    :return: ingest, data_path, re_fit, weights_name, output_name
    """
    ingest = True  # keep this true by default just in case.
    data_path = "F:\\Winter2021\\PythonMusicGenerator\\dataset\\"
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


def read_midi(file):
    print("Loading Music File:", file)

    notes = []
    notes_to_parse = None

    # parse midi file
    midi = m21.converter.parse(file)

    # grouping based on diff instruments
    s2 = m21.instrument.partitionByInstrument(midi)

    # Looping over instr
    for part in s2.parts:
        if 'Piano' in str(part):
            notes_to_parse = part.recurse()

            # note or chord
            for element in notes_to_parse:
                # note
                if isinstance(element, m21.note.Note):
                    notes.append(str(element.pitch))

                # chord
                elif isinstance(element, m21.chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)


def ingest_to_csv(path, num_timesteps, graph):
    """
    knowing that you only want specific numpy arrays, this ingest function loads all midi files
    and saves specifically the ones that need to be used. This means that we can call ingest only
    when we have new data to handle
    :str path: full file path
    :bool graph: whether or not you want to see data
    :return: null
    """
    # integer threshold for frequent notes. Can be changed / edited by looking at graph.
    thresh = 50

    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    notes_array = np.array([read_midi(path + i) for i in files], dtype=object)
    notes_ = [element for note_ in notes_array for element in note_]

    unique_notes = list(set(notes_))  # list of unique notes

    freq = dict(Counter(notes_))
    num = [count for _, count in freq.items()]

    if graph:  # only do this if you want to graph
        plot.figure(figsize=(5, 5))
        plot.hist(num)
        plot.show()

    frequent_notes = [note_ for note_, count in freq.items() if count >= thresh]

    # knowing top freq notes, prepare musical files
    new_music = []
    for notes in notes_array:
        temp = []
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)
            new_music.append(temp)
    new_music = np.array(new_music, dtype=object)

    # now onto prepping data.
    x = []
    y = []

    for note_ in new_music:
        for i in range(0, len(note_) - num_timesteps, 1):
            # print("note_ i:")
            # print("note_ i: ", i)
            # prep input/outpu seqs
            input_ = note_[i:i + num_timesteps]
            output = note_[i + num_timesteps]

            x.append(input_)
            y.append(output)
    print("arrays appended")
    x = np.array(x)
    y = np.array(y)  # bc np arrays >>>>>

    print("x/y arrays made")
    # asign intger to every note
    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

    # prepare integer sequences for input data
    x_seq = []
    for i in x:
        temp = []
        for j in i:
            # assign int to note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)
    x_seq = np.array(x_seq)

    # prepareinteger sequences for output data
    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    y_seq = np.array([y_note_to_int[i] for i in y])
    print("prepped")

    # we want to save y_seq and x_seq for later.
    np.save("x_seq", x_seq)
    np.save("x_unique", unique_x)
    np.save("y_seq", y_seq)
    np.save("y_unique", unique_y)


def fit_model(x_tr, x_val, y_tr, y_val):
    # xx% for training and 100-xx% for evaluation.
    '''currently using 20% for evaluation'''
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
    x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

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

    model.add(klayers.GlobalMaxPool1D())

    model.add(klayers.Dense(256, activation='relu'))
    model.add(klayers.Dense(len(unique_y), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()

    # callback to save best model during training
    # TODO: remember to change the names
    mc = kcallbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
    history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=128, epochs=5,
                        validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])


def convert_to_midi(prediction_output):
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
    midi_stream.write('midi', fp='predicted.mid')


def cov_midi_to_wav(name):
    """
    converts midi file to wav.
    :param name: name of midi file without extension.
    :return: null
    """
    midi_name = name + '.mid'
    output_name = name + '.wav'
    FluidSynth.midi_to_audio(midi_name, output_name)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # # read_midi('F:\\Winter2021\\PythonMusicAI\\dataset\\schu_143_1.mid')
    path = 'F:\\Winter2021\\PythonMusicGenerator\\dataset\\'
    num_timesteps = 32  # you need to change input length if you change this too
    # # path = 'dataset/'

    ingest = True  # do you want to ingest?
    re_fit = True  # do you want to re-fit the model?
    output = 'predicted_sch_15'  # what would you like your output name to be?

    # ingest, path, re_fit, weights_name, output_name = check_inputs()
    # ingest = true if want ingest, false if no need to ingest
    # path = dataset path
    # re_fit = true if model needs to be fitted again, false if already fit.
    # weights_name = checkpoint model name
    # output_name = output of predictions.

    if ingest:
        ingest_to_csv(path, num_timesteps, False)
        print("ingested")

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

    x_seq = np.load("x_seq.npy")
    y_seq = np.load("y_seq.npy")
    unique_x = np.load("x_unique.npy")
    unique_y = np.load("y_unique.npy")
    print("loaded")

    # xx% for training and 100-xx% for evaluation.
    '''currently using 20% for evaluation'''
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # this is not a proper fix.
    x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

    # # start building the model now. Clean this up later.
    # kbackend.clear_session()
    # model = kmodels.Sequential();
    #
    # # embedding layers
    # model.add(klayers.Embedding(len(unique_x), 100, input_length=32, trainable=True))
    #
    # model.add(klayers.Conv1D(64, 3, padding='causal', activation='relu'))
    # model.add(klayers.Dropout(0.2))
    # model.add(klayers.MaxPool1D(2))
    #
    # model.add(klayers.Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
    # model.add(klayers.Dropout(0.2))
    # model.add(klayers.MaxPool1D(2))
    #
    # model.add(klayers.Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
    # model.add(klayers.Dropout(0.2))
    # model.add(klayers.MaxPool1D(2))
    #
    # model.add(klayers.GlobalMaxPool1D())
    #
    # model.add(klayers.Dense(256, activation='relu'))
    # model.add(klayers.Dense(len(unique_y), activation='softmax'))
    #
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    #
    # model.summary()
    #
    # # callback to save best model during training
    # # TODO: remember to change the names
    # mc = kcallbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    #
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #this is not a proper fix.
    # if re_fit:
    #     history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=128, epochs=2,
    #                         validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])

    if re_fit:
        fit_model(train_test_split(x_seq, y_seq, test_size=0.2, random_state=0))
    model = kmodels.load_model('best_model.h5')

# now we compose our own music......
ind = np.random.randint(0, len(x_val) - 1)

random_music = x_val[ind]

predictions = []
for i in range(50):
    random_music = random_music.reshape(1, num_timesteps)

    prob = model.predict(random_music)[0]
    y_pred = np.argmax(prob, axis=0)
    predictions.append(y_pred)

    random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
    random_music = random_music[1:]
print(predictions)

x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
predicted_notes = [x_int_to_note[i] for i in predictions]

convert_to_midi(predicted_notes)
# FluidSynth.midi_to_audio('predicted.mid', 'output.wav')  # hopefully this generates a wav file.
