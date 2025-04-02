{{ config(materialized='view') }}

WITH REVIEW_DATA AS (
    SELECT REVIEW_ID,
    PLACE_ID,
    RATING,
    PUBLISHTIME,
    TEXT
    FROM {{ ref("sushi_place_reviews_fact") }}
),

V_ALL_REVIEW_INFO AS (
SELECT r.*,
       s.WORD_TOKENS,
FROM {{ source('scored_reviews', 'SUSHI_REVIEW_SENTIMENT_FACT') }} s
INNER JOIN REVIEW_DATA r
ON s.REVIEW_ID = r.REVIEW_ID
)

SELECT *
FROM V_ALL_REVIEW_INFO