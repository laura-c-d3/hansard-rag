DELETE FROM feedback WHERE conversation_id IN (SELECT id FROM conversations WHERE question LIKE 'dummy %');
DELETE FROM conversations WHERE question LIKE 'dummy %';