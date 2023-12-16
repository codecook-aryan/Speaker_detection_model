from flask import Flask, render_template, request
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from sklearn import cluster

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', speaker_changes=[])

@app.route('/run_diarization', methods=['POST'])
def run_diarization():
    try:
        audio_file = request.files['audio_file']

        if audio_file and audio_file.filename.endswith('.wav'):
            # Save the uploaded audio file to disk
            file_path = 'uploaded_file.wav'  # Change the file path as needed
            audio_file.save(file_path)

            # Perform speaker diarization on the uploaded file
            (rate, sig) = wav.read(file_path)
            data = mfcc(sig, rate)
            data = data.reshape(data.shape[0], 13)

            # Replace these slicing indices with your logic
            start_index = 179989
            end_index = 179999
            sliced_data = data[start_index:end_index]

            if sliced_data.shape[0] > 0:
                ks = range(1, 2)
                # Explicitly set n_init to suppress FutureWarning
                KMeans = [cluster.KMeans(n_clusters=i, init="k-means++", n_init=10).fit(sliced_data) for i in ks]

                # Your speaker diarization code for identifying speakers here...
                # Example logic for detected speaker changes (replace with your actual logic)
                speaker_changes = []
                for i in range(176915, 176939):
                    speaker_changes.append(f"Speaker change detected at frame {i}")

            else:
                speaker_changes = ['Sliced data is empty. Check slicing indices.']

        else:
            speaker_changes = ['Please upload a valid WAV file.']

    except Exception as e:
        speaker_changes = [f'An error occurred: {str(e)}']

    return render_template('index.html', speaker_changes=speaker_changes)

if __name__ == '__main__':
    app.run(debug=True)
