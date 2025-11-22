
-- Run after creating tables/extensions. Adjust schema/paths as needed.

-- Load users
\copy app_user(id, created_at, active, dob, gender, orientation, bio, photos, location, last_active_at)
FROM PROGRAM 'awk -F, "NR>1{printf \"%s,%s,%s,%s,%s,{%s},%s,%s,SRID=4326;POINT(%s %s),%s\n\", $1,$2,$3,$4,$5,$6,$7,$8,$10,$9,$11}" /mnt/data/app_user.csv'
WITH (FORMAT csv);

-- Alternative: use a temp table to assemble geography point from lat/lon if awk is unavailable.

-- Load prefs
\copy user_pref(user_id, age_min, age_max, max_distance_km, genders, orientations)
FROM PROGRAM 'awk -F, "NR>1{printf \"%s,%s,%s,%s,{%s},{%s}\n\", $1,$2,$3,$4,$5,$6}" /mnt/data/user_pref.csv'
WITH (FORMAT csv);

-- Load embeddings (JSON -> vector via explicit cast)
-- If using pgvector 0.6+, you can use to_vector; otherwise transform client-side.
-- Here we'll use a staging table then INSERT ... SELECT.

CREATE TEMP TABLE _emb_stage(user_id bigint, embedding_json jsonb, updated_at timestamptz);
\copy _emb_stage(user_id, embedding_json, updated_at) FROM /mnt/data/profile_embedding.csv CSV HEADER;

INSERT INTO profile_embedding(user_id, embedding, updated_at)
SELECT user_id, to_vector(embedding_json)::vector(128), updated_at
FROM _emb_stage;

-- Load metrics
\copy profile_metrics(user_id, views_7d, likes_7d, last_seen_at) FROM /mnt/data/profile_metrics.csv CSV HEADER;

-- Load interactions
\copy interaction(actor_id, target_id, type, created_at) FROM /mnt/data/interaction.csv CSV HEADER;
