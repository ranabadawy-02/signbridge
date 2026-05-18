# -*- coding: utf-8 -*-
"""
AI SERVER - Sign Language Recognition API
Receives frames from the browser, runs MSTP model, returns Arabic text.
Run this alongside your Node.js server.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import mediapipe as mp
try:
    from mediapipe.python.solutions import holistic as mp_holistic_module
    from mediapipe.python.solutions import drawing_utils as mp_drawing_module
    _NEW_MP = True
except ImportError:
    _NEW_MP = False
from collections import deque, Counter
import base64
import re
import os
import asyncio
import edge_tts
import tempfile
import time  # ← NEW: needed for time-based frame gap detection

app = Flask(__name__)
CORS(app)

# ==============================================================================
# ARABIC PRONUNCIATION MAP
# ==============================================================================
EGYPTIAN_TTS_MAP = {
    "ايه": "إي",
    "ازيك": "إزَيَّكْ",
    "مش": "مِشْ",
    "مصر": "مَصْرْ",
    "منين": "مِنينْ",
    "انت": "إنتَ",
    "مهاراتك": "مهاراتَكْ",
    "اسمك": "اسمَك",
    "تعيد": "تِعيدْ",
    "تانى": "تانيي",
    "اتشرفت": "إتشَرّفتْ",
    "الحمد": "الحمدُ",
    "بمعرفتك": "بِمَعْرِفْتَكْ",
}

def normalize_arabic(word):
    word = word.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    word = word.replace("ة", "ه")
    return word

def add_sukun(word):
    if word.endswith("ْ"):
        return word
    return word + "ْ"

def convert_to_egyptian_pronunciation(text):
    words = text.split()
    converted_words = []
    for word in words:
        clean_word = re.sub(r'[^\u0600-\u06FF]', '', word)
        normalized_word = normalize_arabic(clean_word)
        if normalized_word in EGYPTIAN_TTS_MAP:
            spoken = EGYPTIAN_TTS_MAP[normalized_word]
        else:
            spoken = add_sukun(clean_word)
        converted_words.append(spoken)
    return " ".join(converted_words)

# ==============================================================================
# EDGE TTS
# ==============================================================================
VOICE_OPTIONS = {
    "male":   "ar-EG-ShakirNeural",
    "female": "ar-EG-SalmaNeural",
}

def generate_tts_audio(text, voice="female", rate="+0%", volume="+0%", pitch="+0Hz"):
    async def _gen():
        voice_name = VOICE_OPTIONS.get(voice, VOICE_OPTIONS["female"])
        communicate = edge_tts.Communicate(text, voice_name, rate=rate,
                                           volume=volume, pitch=pitch)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        os.unlink(tmp_path)
        return base64.b64encode(audio_data).decode("utf-8")

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_gen())
        loop.close()
        return result
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return None

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class MS_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_branches=4):
        super(MS_TCN, self).__init__()
        self.num_branches = num_branches
        self.dim_reduction = nn.Conv1d(in_channels, out_channels // num_branches,
                                       kernel_size=1)
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            dilation = i + 1
            branch = nn.Sequential(
                nn.Conv1d(out_channels // num_branches,
                          out_channels // num_branches,
                          kernel_size=3, padding=dilation, dilation=dilation),
                nn.BatchNorm1d(out_channels // num_branches),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dim_reduction(x)
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        out = self.final_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class TemporalLiftPooling(nn.Module):
    def __init__(self):
        super(TemporalLiftPooling, self).__init__()

    def forward(self, x):
        B, C, T = x.shape
        if T % 2 == 1:
            x = F.pad(x, (0, 1), mode='replicate')
            T += 1
        x_even = x[:, :, 0::2]
        x_odd  = x[:, :, 1::2]
        return (x_even + x_odd) / 2.0


class MSTP_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(MSTP_Module, self).__init__()
        self.ms_tcn = MS_TCN(input_dim, hidden_dim, num_branches=4)
        self.conv1  = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm1d(hidden_dim)
        self.relu1  = nn.ReLU(inplace=True)
        self.tlp1   = TemporalLiftPooling()
        self.blstm  = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=num_layers,
                              bidirectional=True, batch_first=True)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.ms_tcn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.tlp1(x)
        if lengths is not None:
            lengths = (lengths + 1) // 2
        x = x.transpose(1, 2)
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        x, _ = self.blstm(x)
        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        return x, lengths


class SwinMSTP(nn.Module):
    def __init__(self, input_dim=132, hidden_dim=128, vocab_size=18,
                 num_layers=2, dropout=0.2):
        super(SwinMSTP, self).__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mstp           = MSTP_Module(hidden_dim, hidden_dim, num_layers=num_layers)
        self.classifier     = nn.Linear(hidden_dim, vocab_size)
        self.aux_classifier = nn.Linear(hidden_dim, vocab_size)
        with torch.no_grad():
            self.classifier.bias[0]     = -2.0
            self.aux_classifier.bias[0] = -2.0
        self.aux_classifier.weight = self.classifier.weight
        self.aux_classifier.bias   = self.classifier.bias

    def forward(self, features, lengths, use_aux=False):
        x = self.input_projection(features)
        x, out_lengths = self.mstp(x, lengths)
        logits    = (self.aux_classifier(x) if use_aux and self.training
                     else self.classifier(x))
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, out_lengths

# ==============================================================================
# NORMALIZATION
# ==============================================================================
def normalize_skeleton(data_75):
    left_shoulder  = data_75[11]
    right_shoulder = data_75[12]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width == 0:
        shoulder_width = 1.0
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    lh = data_75[33:54]
    rh = data_75[54:75]
    if np.any(lh):
        lh_wrist = lh[0]
        lh_norm  = (lh - lh_wrist) / shoulder_width
        lh_ref   = (lh_wrist - mid_shoulder) / shoulder_width
    else:
        lh_norm = np.zeros((21, 3))
        lh_ref  = np.zeros(3)
    if np.any(rh):
        rh_wrist = rh[0]
        rh_norm  = (rh - rh_wrist) / shoulder_width
        rh_ref   = (rh_wrist - mid_shoulder) / shoulder_width
    else:
        rh_norm = np.zeros((21, 3))
        rh_ref  = np.zeros(3)
    return np.concatenate([lh_norm.flatten(), rh_norm.flatten(), lh_ref, rh_ref])


def extract_raw_landmarks(results):
    pose = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.pose_landmarks.landmark])
            if results.pose_landmarks else np.zeros((33, 3)))
    lh   = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.left_hand_landmarks.landmark])
            if results.left_hand_landmarks else np.zeros((21, 3)))
    rh   = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.right_hand_landmarks.landmark])
            if results.right_hand_landmarks else np.zeros((21, 3)))
    return np.concatenate([pose, lh, rh])

# ==============================================================================
# GLOBAL STATE (per peer session)
# ==============================================================================
peer_states = {}

def get_peer_state(peer_id):
    if peer_id not in peer_states:
        peer_states[peer_id] = {
            # FIX ④: reduced from 120 → 60 so buffer doesn't hold too much
            # history at slow browser FPS (~10–15 FPS over HTTP)
            "frames_buffer":         deque(maxlen=60),
            "hand_detection_buffer": deque(maxlen=5),
            "prediction_history":    deque(maxlen=5),
            "motion_buffer":         deque(maxlen=15),
            "prev_hand_positions":   None,
            "is_currently_signing":  False,
            "sign_start_frame":      0,
            "last_prediction":       None,
            "rest_counter":          0,
            "frame_count":           0,
            "stable_prediction":     None,
            "current_confidence":    0.0,
            "is_moving":             False,
            "no_hand_counter":       0,
            # FIX ①②③: time-based frame tracking
            "last_frame_time":       None,
            "frame_interval_ms":     33.0,   # assume 30 FPS until measured
        }
    return peer_states[peer_id]


def full_reset_sign_state(state):
    """
    Full clean reset between signs.
    Clears ALL motion/buffer state so the next sign starts completely fresh.
    Prevents leftover motion from the previous sign contaminating the next one.
    """
    state["frames_buffer"].clear()
    state["prediction_history"].clear()
    state["motion_buffer"].clear()
    state["prev_hand_positions"] = None
    state["last_prediction"]     = None
    state["rest_counter"]        = 0
    state["is_moving"]           = False
    state["current_confidence"]  = 0.0

# ==============================================================================
# LOAD MODEL ON STARTUP
# ==============================================================================
DEVICE     = torch.device('cpu')
MODEL_PATH = 'best_18_sentence_mstp.pth'
model      = None
vocab      = None
idx2gloss  = None
holistic   = None

if _NEW_MP:
    mp_holistic = mp_holistic_module
else:
    mp_holistic = mp.solutions.holistic

def load_model():
    global model, vocab, idx2gloss, holistic
    print("📂 Loading model...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        vocab      = checkpoint['vocab']
        idx2gloss  = {v: k for k, v in vocab.items()}
        vocab_size = len(vocab)
        model = SwinMSTP(input_dim=132, hidden_dim=128, vocab_size=vocab_size,
                         num_layers=2, dropout=0.2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✅ Model loaded! Vocab size: {vocab_size}")
    except FileNotFoundError:
        print(f"❌ Model not found at {MODEL_PATH}.")

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    print("✅ MediaPipe ready!")

# ==============================================================================
# THRESHOLDS
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.25
MOTION_THRESHOLD     = 0.003   # normalized to 30 FPS baseline
# FIX ③: time-based rest instead of frame-count based
REST_DURATION_MS     = 800     # 200ms of stillness ends a sign
MIN_SIGN_DURATION    = 3     # minimum frames before accepting a sign
# FIX ①: maximum gap between frames before resetting state
MAX_FRAME_GAP_MS     = 2000    # if >200ms since last frame, state is stale

# ==============================================================================
# MAIN INFERENCE ENDPOINT
# ==============================================================================
@app.route('/process_frame', methods=['POST'])
def process_frame():
    if model is None or holistic is None:
        return jsonify({"error": "Model not loaded"}), 503

    data       = request.get_json()
    peer_id    = data.get("peerId",  "unknown")
    frame_b64  = data.get("frame",   "")
    tts_voice  = data.get("voice",   "female")
    tts_rate   = data.get("rate",    "+0%")
    tts_volume = data.get("volume",  "+0%")
    tts_pitch  = data.get("pitch",   "+0Hz")

    # ------------------------------------------------------------------
    # Decode frame
    # Browser canvas toDataURL() gives correct orientation already.
    # FIX ①: supports both PNG (lossless) and JPEG from the frontend.
    # Prefer PNG on the frontend: canvas.toDataURL("image/png")
    # ------------------------------------------------------------------
    try:
        img_bytes = base64.b64decode(frame_b64.split(",")[-1])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400
    except Exception as e:
        return jsonify({"error": f"Frame decode error: {str(e)}"}), 400

    # ------------------------------------------------------------------
    # Per-peer state
    # ------------------------------------------------------------------
    state = get_peer_state(peer_id)
    state["frame_count"] += 1
    frame_count = state["frame_count"]

    # FIX ②: measure actual frame interval
    now = time.time()
    if state["last_frame_time"] is not None:
        measured_ms = (now - state["last_frame_time"]) * 1000.0
        # Smooth the interval with an EMA to avoid spikes
        state["frame_interval_ms"] = (
            0.7 * state["frame_interval_ms"] + 0.3 * measured_ms
        )
        # FIX ②: if there was a large gap, the buffer context is stale — reset
        if measured_ms > MAX_FRAME_GAP_MS:
            print(f"⚠️  Large frame gap ({measured_ms:.0f}ms) for peer {peer_id}, resetting state")
            full_reset_sign_state(state)
    state["last_frame_time"] = now

    # ------------------------------------------------------------------
    # MediaPipe
    # ------------------------------------------------------------------
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    hands_visible = (results.left_hand_landmarks  is not None or
                     results.right_hand_landmarks is not None)

    # Reset if hands gone too long
    if not hands_visible:
        state["no_hand_counter"] += 1
        if state["no_hand_counter"] >= 25:
            state["is_currently_signing"] = False
            full_reset_sign_state(state)
            state["stable_prediction"] = None
    else:
        state["no_hand_counter"] = 0

    state["hand_detection_buffer"].append(hands_visible)

    hands_stable   = sum(state["hand_detection_buffer"]) >= 2
    detection_rate = (sum(state["hand_detection_buffer"]) /
                      len(state["hand_detection_buffer"])
                      if state["hand_detection_buffer"] else 0)
    good_detection = detection_rate > 0.5

    # ------------------------------------------------------------------
    # FIX ③: Motion detection normalized by frame interval
    # Dividing by actual elapsed time keeps the threshold consistent
    # regardless of whether frames arrive at 10 FPS or 30 FPS.
    # ------------------------------------------------------------------
    current_hand_positions = []
    if results.left_hand_landmarks:
        current_hand_positions.append([
            results.left_hand_landmarks.landmark[0].x,
            results.left_hand_landmarks.landmark[0].y
        ])
    if results.right_hand_landmarks:
        current_hand_positions.append([
            results.right_hand_landmarks.landmark[0].x,
            results.right_hand_landmarks.landmark[0].y
        ])

    if state["prev_hand_positions"] and current_hand_positions:
        total_movement = 0
        for prev_pos, curr_pos in zip(state["prev_hand_positions"],
                                      current_hand_positions):
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            total_movement += (dx**2 + dy**2) ** 0.5
        avg_movement = total_movement / len(current_hand_positions)

        # Normalize to 33ms (30 FPS baseline) so threshold stays consistent
        interval_ms = max(state["frame_interval_ms"], 10.0)
        normalized_movement = avg_movement * (33.0 / interval_ms)

        state["motion_buffer"].append(normalized_movement)
        recent_motion      = (sum(state["motion_buffer"]) /
                              len(state["motion_buffer"])
                              if state["motion_buffer"] else 0)
        state["is_moving"] = recent_motion > MOTION_THRESHOLD
        print(f"🖐️ recent_motion={recent_motion:.6f} threshold={MOTION_THRESHOLD} is_moving={state['is_moving']}")
    if current_hand_positions:
        state["prev_hand_positions"] = current_hand_positions

    # ------------------------------------------------------------------
    # Landmarks → features
    # ------------------------------------------------------------------
    raw_landmarks       = extract_raw_landmarks(results)
    normalized_features = normalize_skeleton(raw_landmarks)
    state["frames_buffer"].append(normalized_features)

    # ------------------------------------------------------------------
    # Sign segmentation
    # FIX ③: rest detection is now time-based (REST_DURATION_MS),
    # not frame-count based, so it works correctly at any FPS.
    # ------------------------------------------------------------------
    result_text  = None
    result_tts   = None
    result_audio = None

    # FIX ③: compute rest duration in milliseconds
    interval_ms  = max(state["frame_interval_ms"], 10.0)
    rest_duration_ms = state["rest_counter"] * interval_ms

    # Force-end stuck signs (still frame-count based as a safety cap)
    MAX_SIGN_FRAMES = 80
    if state["is_currently_signing"]:
        if (frame_count - state["sign_start_frame"]) > MAX_SIGN_FRAMES:
            state["is_currently_signing"] = False
            if state["last_prediction"]:
                state["stable_prediction"] = state["last_prediction"]
            full_reset_sign_state(state)

    if state["is_moving"]:
        state["rest_counter"] = 0
        if not state["is_currently_signing"]:
            state["is_currently_signing"] = True
            state["sign_start_frame"]     = frame_count
            state["frames_buffer"].clear()
            state["prediction_history"].clear()
            state["stable_prediction"] = None
    else:
        state["rest_counter"] += 1

        # Early prediction lock — fire immediately if very confident
        if (state["is_currently_signing"]
                and state["current_confidence"] > 0.60
                and state["last_prediction"]):
            state["is_currently_signing"] = False
            state["stable_prediction"]    = state["last_prediction"]
            result_text  = state["stable_prediction"]
            result_tts   = convert_to_egyptian_pronunciation(result_text)
            print(f"⚡ EARLY LOCK: {result_text} → TTS: {result_tts}")
            result_audio = generate_tts_audio(result_tts, tts_voice,
                                              tts_rate, tts_volume, tts_pitch)
            full_reset_sign_state(state)

        # FIX ③: use time-based rest (REST_DURATION_MS) instead of frame count
        elif (state["is_currently_signing"]
                and rest_duration_ms >= REST_DURATION_MS):
            state["is_currently_signing"] = False
            sign_duration = frame_count - state["sign_start_frame"]

            if sign_duration >= MIN_SIGN_DURATION and state["last_prediction"]:
                state["stable_prediction"] = state["last_prediction"]
                result_text  = state["stable_prediction"]
                result_tts   = convert_to_egyptian_pronunciation(result_text)
                print(f"✅ SIGN: {result_text} → TTS: {result_tts}")
                result_audio = generate_tts_audio(result_tts, tts_voice,
                                                  tts_rate, tts_volume, tts_pitch)

            full_reset_sign_state(state)

    # ------------------------------------------------------------------
    # Model inference — every 2nd frame, min 10 frames
    # ------------------------------------------------------------------
    if (len(state["frames_buffer"]) >= 3
        and state["is_currently_signing"]
        and good_detection):
        try:
            features        = np.stack(list(state["frames_buffer"]), axis=0)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            lengths         = torch.LongTensor([len(state["frames_buffer"])])

            mean            = features_tensor.mean(dim=1, keepdim=True)
            std             = features_tensor.std(dim=1, keepdim=True) + 1e-8
            features_tensor = (features_tensor - mean) / std

            with torch.no_grad():
                log_probs, out_lengths = model(features_tensor, lengths,
                                               use_aux=False)

            # CTC decode
            predictions  = []
            prev_idx     = None
            logits       = log_probs[0, :out_lengths[0], :]
            pred_indices = torch.argmax(logits, dim=-1).numpy()
            for idx in pred_indices:
                if idx != 0 and idx != prev_idx:
                    gloss = idx2gloss.get(idx, '<UNK>')
                    if gloss not in ['<PAD>', '<UNK>']:
                        predictions.append(gloss)
                prev_idx = idx

            probs = torch.exp(log_probs[0, :out_lengths[0], :])
            state["current_confidence"] = (torch.max(probs, dim=-1)[0]
                                           .mean().item())

            if (predictions and
                    state["current_confidence"] > CONFIDENCE_THRESHOLD):
                pred_text    = ' '.join(predictions)
                state["prediction_history"].append(pred_text)
                vote_counter = Counter(state["prediction_history"])
                most_common  = vote_counter.most_common(1)[0][0]
                state["last_prediction"] = most_common

        except Exception as e:
            print(f"❌ Inference error: {e}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    if state["is_currently_signing"]:
        status = "signing"
    elif not hands_visible:
        status = "no_hands"
    elif state["stable_prediction"]:
        status = "complete"
    else:
        status = "ready"

    return jsonify({
        "text":       result_text,
        "tts_text":   result_tts,
        "audio":      result_audio,
        "status":     status,
        "confidence": round(state["current_confidence"] * 100, 1),
        "is_signing": state["is_currently_signing"],
    })


@app.route('/reset_peer', methods=['POST'])
def reset_peer():
    data    = request.get_json()
    peer_id = data.get("peerId", "")
    if peer_id in peer_states:
        del peer_states[peer_id]
    return jsonify({"ok": True})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
        "active_peers": len(peer_states)
    })


if __name__ == '__main__':
    load_model()
    print(f"\n🚀 AI Server running on port 5001")
    print(f"📦 Model path: {MODEL_PATH}\n")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=False)
