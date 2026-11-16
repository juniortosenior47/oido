CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE audio_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    source VARCHAR(80) NOT NULL DEFAULT 'live_microphone',
    sample_rate INTEGER NOT NULL,
    channels SMALLINT NOT NULL DEFAULT 1,
    language VARCHAR(16) NOT NULL DEFAULT 'es',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE audio_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES audio_sessions(id) ON DELETE CASCADE,
    sequence_number INTEGER NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    rms_dbfs REAL,
    spectral_centroid_hz REAL,
    audio_path TEXT,
    dsp_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE(session_id, sequence_number)
);

CREATE TABLE transcriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES audio_segments(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    predicted_text TEXT NOT NULL,
    language VARCHAR(16) NOT NULL DEFAULT 'es',
    model_name VARCHAR(120) NOT NULL,
    model_version VARCHAR(80),
    confidence REAL,
    is_final BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE(segment_id)
);

CREATE TABLE corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcription_id UUID NOT NULL UNIQUE REFERENCES transcriptions(id) ON DELETE CASCADE,
    corrected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    corrected_text TEXT NOT NULL,
    corrected_by VARCHAR(120) NOT NULL DEFAULT 'operator',
    notes TEXT
);

-- Whisper y otros motores externos son validadores; la transcripción primaria
-- siempre pertenece a oido-asr-core.
CREATE TABLE validator_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES audio_segments(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    validator_name VARCHAR(120) NOT NULL,
    validator_version VARCHAR(80),
    predicted_text TEXT NOT NULL,
    confidence REAL,
    UNIQUE(segment_id, validator_name)
);

CREATE TABLE comparison_rounds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transcription_id UUID NOT NULL REFERENCES transcriptions(id) ON DELETE CASCADE,
    validator_prediction_id UUID REFERENCES validator_predictions(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    similarity REAL NOT NULL CHECK (similarity >= 0 AND similarity <= 1),
    game_state VARCHAR(2) NOT NULL CHECK (game_state IN ('CC', 'CD', 'DC', 'DD')),
    own_confidence REAL,
    validator_confidence REAL,
    is_human_verified BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(transcription_id, validator_prediction_id)
);

CREATE TABLE learning_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correction_id UUID NOT NULL UNIQUE REFERENCES corrections(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    predicted_text TEXT NOT NULL,
    reference_text TEXT NOT NULL,
    similarity REAL NOT NULL CHECK (similarity >= 0 AND similarity <= 1),
    confidence REAL,
    model_action VARCHAR(16) NOT NULL CHECK (model_action IN ('cooperate', 'defect')),
    reference_action VARCHAR(16) NOT NULL CHECK (reference_action IN ('cooperate', 'defect')),
    payoff REAL NOT NULL,
    reward REAL NOT NULL,
    weight_delta JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_version VARCHAR(64) NOT NULL
);

CREATE TABLE weight_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    policy_version VARCHAR(64) NOT NULL,
    weights JSONB NOT NULL,
    evaluation JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE word_labels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE oaf_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label_id UUID NOT NULL REFERENCES word_labels(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    feature_version VARCHAR(32) NOT NULL,
    feature_path TEXT NOT NULL,
    frame_count INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
    notes TEXT
);

CREATE TABLE oaf_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    feature_version VARCHAR(32) NOT NULL, sample_count INTEGER NOT NULL,
    parameters JSONB NOT NULL, metrics JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE oaf_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), training_run_id UUID NOT NULL REFERENCES oaf_training_runs(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), model_path TEXT NOT NULL, weights JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE oaf_recognition_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_id UUID REFERENCES oaf_models(id), predicted_label TEXT, confidence REAL,
    feature_path TEXT, reviewed_label TEXT
);

CREATE INDEX idx_audio_segments_session ON audio_segments(session_id, sequence_number);
CREATE INDEX idx_transcriptions_created ON transcriptions(created_at DESC);
CREATE INDEX idx_learning_events_created ON learning_events(created_at DESC);
CREATE INDEX idx_comparison_rounds_state ON comparison_rounds(game_state, created_at DESC);

INSERT INTO weight_snapshots (policy_version, weights, evaluation)
SELECT 'pd-v1', '{"similarity": 1.0, "confidence": 0.4, "history": 0.2}', '{"description": "Pesos iniciales de decisión, no pesos neuronales"}'::jsonb
WHERE NOT EXISTS (SELECT 1 FROM weight_snapshots);
