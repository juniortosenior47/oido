-- Ejecute este archivo una vez si PostgreSQL ya se creó con schema.sql previo.
CREATE TABLE IF NOT EXISTS validator_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES audio_segments(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    validator_name VARCHAR(120) NOT NULL,
    validator_version VARCHAR(80),
    predicted_text TEXT NOT NULL,
    confidence REAL,
    UNIQUE(segment_id, validator_name)
);

CREATE TABLE IF NOT EXISTS comparison_rounds (
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
CREATE INDEX IF NOT EXISTS idx_comparison_rounds_state ON comparison_rounds(game_state, created_at DESC);
