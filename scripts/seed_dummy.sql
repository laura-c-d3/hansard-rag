INSERT INTO conversations (id, ts, question, answer, route, latency_s, prompt_tokens, completion_tokens, n_sources)
SELECT
    gen_random_uuid(),
    now() - (random() * interval '48 hours'),
    'dummy question ' || g,
    'dummy answer ' || g,
    (ARRAY['SEARCH','SEARCH','SEARCH','DEBATE_SUMMARY','SPEAKER'])[1 + floor(random()*5)],
    round((1.5 + random() * 6)::numeric, 2),
    (1500 + floor(random() * 4000))::int,
    (150 + floor(random() * 500))::int,
    (3 + floor(random() * 5))::int
FROM generate_series(1, 120) AS g;

-- feedback for some responses
INSERT INTO feedback (conversation_id, ts, thumbs_up)
SELECT id, ts + interval '30 seconds', random() < 0.8
FROM conversations
WHERE question LIKE 'dummy %' AND random() < 0.5;