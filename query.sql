SELECT r.review_id, r.text
FROM `yelp-ra-work-396117.Yelp_review_text.reviews` r
LEFT JOIN `yelp-ra-work-396117.models_pca.philadelphia_review_embeddings` t1 
ON r.review_id = t1.review_id
LEFT JOIN `yelp-ra-work-396117.models_pca.saint_louis_review_embeddings` t2 
ON r.review_id = t2.review_id
WHERE t1.review_id IS NULL 
AND t2.review_id IS NULL
AND r.review_id > "vSJ20XNnwpss8jQrpuuptA"
ORDER BY r.review_id
