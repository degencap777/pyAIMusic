from midi2audio import FluidSynth
import os
import music21 as m21
from collections import Counter
import matplotlib.pyplot as plot
from progress.bar import Bar, ChargingBar
import numpy as np


def cov_midi_to_wav(name):
    """
    converts midi file to wav.
    :param name: name of midi file without extension.
    :return: null
    """
    midi_name = name + '.mid'
    output_name = name + '.wav'
    FluidSynth.midi_to_audio(midi_name, output_name)


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


def longseq_ingest(path, num_timesteps=32, thresh=50, graph=True):
    """
    long sequential ingest (no batching).
    :path: filepath to directory with input midi files
    :num_timesteps: number of timesteps (default 32)
    :thresh: threshold of frequent notes to bar (default 50)
    :graph: graph note-frequencies (default True)
    :return:
    """

    files = [i for i in os.listdir(path) if i.endswith(".mid")]
    notes_array = np.array([read_midi(path + i) for i in files], dtype=object)
    notes_ = [element for note_ in notes_array for element in note_]
    freq = dict(Counter(notes_))
    num = [count for _, count in freq.items()]

    if graph:  # only do this if you want to graph
        plot.figure(figsize=(5, 5))
        plot.hist(num)
        plot.show()

    print("applying frequent notes")

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

    del frequent_notes # memory optimizations
    del notes_array
    del notes_

    print("Done calculating frequent notes. \n Prepping data.")

    # now onto prepping data.
    x = []
    y = []
    bar_note_ = ChargingBar("Collapsing note arrays", max=len(new_music))
    spinner_bool = True
    for note_ in new_music:
        # bar_range = Bar("inner i", max=len(note_), color='cyan')
        for i in range(0, len(note_) - num_timesteps, 1):
            input_ = note_[i:i + num_timesteps]
            output = note_[i + num_timesteps]

            x.append(input_)
            y.append(output)
            # bar_range.next()
        # bar_range.finish()
        bar_note_.next()
    bar_note_.finish()
    print("arrays appended")
    del new_music
    x = np.array(x)
    y = np.array(y)  # bc np arrays >>>>>

    print("x/y arrays made")
    # asign intger to every note
    unique_x = list(set(x.ravel()))
    x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))

    # prepare integer sequences for input data
    bar = ChargingBar("appending sequence arrays", max=len(x))
    x_seq = []
    for i in x:
        temp = []
        for j in i:
            # assign int to note
            temp.append(x_note_to_int[j])
        x_seq.append(temp)
        bar.next()
    bar.finish()
    x_seq = np.array(x_seq)
    del x

    # prepareinteger sequences for output data
    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    y_seq = np.array([y_note_to_int[i] for i in y])
    print("prepped")
    del y

    # we want to save y_seq and x_seq for later.
    np.save("x_seq", x_seq)
    np.save("x_unique", unique_x)
    np.save("y_seq", y_seq)
    np.save("y_unique", unique_y)
    print("saved")
    return;
