"""Prompt templates for serve detection and video analysis."""

# System prompt that establishes the model's role and expertise
SYSTEM_PROMPT = """You are an expert pickleball video analyst. Your task is to analyze video frames 
and identify specific game events, particularly serves. You have deep knowledge of pickleball rules, 
court positioning, and player movements."""

# Main serve detection prompt - designed for single frame analysis
SERVE_DETECTION_PROMPT = """Analyze this pickleball video frame and determine if a player is serving.

A pickleball serve has these specific characteristics:
1. **Position**: Server stands behind the baseline (back line of the court)
2. **Motion**: Underhand paddle motion - paddle must contact ball below waist level
3. **Ball state**: Ball is being dropped or has just been hit (not coming from opponent)
4. **Court state**: This is the START of a rally - players are typically at baselines, not at the kitchen (non-volley zone)
5. **Diagonal service**: Server aims diagonally to opponent's service court

Look for these visual cues:
- Player at the back of the court, not near the net
- Arm/paddle in low position (underhand)
- Ball near the server's body or just leaving the paddle
- Other players positioned at or near their baselines

Respond in this exact format:
SERVE_DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation of why this is or is not a serve]
PLAYER_POSITION: [Description of where the potential server is on court, if applicable]"""

# Prompt for analyzing a sequence of frames (temporal context)
SERVE_SEQUENCE_PROMPT = """Analyze this sequence of pickleball video frames to detect a serve.

You are given {num_frames} frames in chronological order. Look for the serve SEQUENCE:
1. Pre-serve: Player stationary at baseline, holding ball
2. Ball drop/toss: Ball released from non-paddle hand  
3. Contact: Paddle strikes ball in underhand motion
4. Follow-through: Ball travels diagonally toward opponent's court

Examine the progression across frames to identify if a serve is occurring.

Respond in this exact format:
SERVE_DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
SERVE_FRAME: [Frame number where serve contact occurs, or N/A]
REASONING: [Explanation based on the sequence of events observed]"""

# Exact string used in ``notebooks/serve_detection_training.ipynb`` (A3 train_data.json).
# LoRA SFT targets YES/NO assistant replies to this user text (no system message in that JSON).
SERVE_TRAINING_STYLE_PROMPT = "Is this a pickleball serve situation? Answer YES or NO."

# Multi-frame prompt used in ``serve_shortclip_finetune_colab.ipynb`` (short-clip LoRA SFT).
SERVE_SHORTCLIP_MULTI_PROMPT = (
    "These frames are in time order from a pickleball video. "
    "Is this a serve? Answer YES or NO only."
)

# Simplified prompt for quick classification (faster inference)
SERVE_QUICK_PROMPT = """Is this a pickleball serve? 
Look for: player at baseline, underhand motion, ball being struck/dropped.
Answer only: YES or NO"""

# Prompt for batch processing multiple frames
SERVE_BATCH_PROMPT = """Analyze each of these {num_frames} pickleball frames for serves.

For each frame, determine if it shows a serve and provide a confidence score.

A serve shows: player at baseline, underhand paddle motion, ball contact or drop.

Respond with a list:
FRAME 1: [YES/NO] - [HIGH/MEDIUM/LOW confidence]
FRAME 2: [YES/NO] - [HIGH/MEDIUM/LOW confidence]
... and so on for each frame."""

# Prompt for extracting detailed game state (future use)
GAME_STATE_PROMPT = """Analyze this pickleball frame and describe the current game state.

Identify:
1. Number of players visible
2. Player positions (baseline, kitchen, transition zone)
3. Ball location (if visible)
4. Current action (serve, rally, between points)
5. Court type (indoor/outdoor) if determinable

Provide a structured analysis."""


def get_serve_detection_prompt(include_system: bool = True) -> str:
    """Get the full serve detection prompt."""
    if include_system:
        return f"{SYSTEM_PROMPT}\n\n{SERVE_DETECTION_PROMPT}"
    return SERVE_DETECTION_PROMPT


def get_serve_training_style_prompt() -> str:
    """Prompt matching ``serve_detection_training.ipynb`` / train_data.json (user message only)."""
    return SERVE_TRAINING_STYLE_PROMPT


def get_shortclip_training_style_prompt() -> str:
    """Multi-image prompt matching short-clip serve LoRA training (Colab notebook)."""
    return SERVE_SHORTCLIP_MULTI_PROMPT


def get_sequence_prompt(num_frames: int) -> str:
    """Get the sequence analysis prompt with frame count."""
    return SERVE_SEQUENCE_PROMPT.format(num_frames=num_frames)


def get_batch_prompt(num_frames: int) -> str:
    """Get the batch processing prompt with frame count."""
    return SERVE_BATCH_PROMPT.format(num_frames=num_frames)
