-- Chunk kind (prose vs. one-chunk-per-table) and the printed page labels of its span.
-- DEFAULT 'text' keeps the column addable to a non-empty table and older inserts valid.
ALTER TABLE chunks
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'text' CHECK (kind IN ('text','table')),
    ADD COLUMN label_start TEXT,
    ADD COLUMN label_end TEXT;
