from midi2audio import FluidSynth

def cov_midi_to_wav(name):
    """
    converts midi file to wav.
    :param name: name of midi file without extension.
    :return: null
    """
    midi_name = name + '.mid'
    output_name = name + '.wav'
    FluidSynth.midi_to_audio(midi_name, output_name)

