{{ config(materialized='table') }}

WITH STD_ADDRESSES AS (
SELECT ID,
       -- Create cases for addresses
       CASE WHEN FORMATTEDADDRESS IN ('645, 108 Rossville Ave, Staten Island, NY 10309, USA',
                                      'Staten Island Mall, 2655 Richmond Ave, Staten Island, NY 10314, USA',
                                      'Muzi Sushi, 2333 Hylan Blvd, Staten Island, NY 10306, USA',
                                      '1775-D, Richmond Rd, Staten Island, NY 10306, USA',
                                      'Inside Bon Cafe et Restaurant, 41-31 Queens Blvd, Sunnyside, NY 11104, USA',
                                      '396 3rd Ave, E 28th St, New York, NY 10016, USA',
                                      'G15/G16, 133-36 37th Ave, Flushing, NY 11354, USA') 
                THEN TRIM(CONCAT(SPLIT_PART(FORMATTEDADDRESS, ',', 1), ', ', SPLIT_PART(FORMATTEDADDRESS, ',', 2)))
            WHEN FORMATTEDADDRESS = 'BYOB (No Corkage Fee), 60 Min Per Session, 104 W Washington Pl, New York, NY 10014, USA' 
                THEN TRIM(SPLIT_PART(FORMATTEDADDRESS, ',', 3))
            WHEN FORMATTEDADDRESS = 'ALCOHOL & BYOB, 90 Min Per Session Please Be On Time, White Store, 464 Bergen St Front, Brooklyn, NY 11217, USA' 
                THEN TRIM(SPLIT_PART(FORMATTEDADDRESS, ',', 4))
            ELSE TRIM(SPLIT_PART(FORMATTEDADDRESS, ',', 1)) END AS ADDRESS,
        -- Create cases for cities
        CASE WHEN FORMATTEDADDRESS IN ('645, 108 Rossville Ave, Staten Island, NY 10309, USA',
                                      'Staten Island Mall, 2655 Richmond Ave, Staten Island, NY 10314, USA',
                                      'Muzi Sushi, 2333 Hylan Blvd, Staten Island, NY 10306, USA',
                                      '1775-D, Richmond Rd, Staten Island, NY 10306, USA',
                                      'Inside Bon Cafe et Restaurant, 41-31 Queens Blvd, Sunnyside, NY 11104, USA',
                                      '396 3rd Ave, E 28th St, New York, NY 10016, USA',
                                      'G15/G16, 133-36 37th Ave, Flushing, NY 11354, USA')
                THEN SPLIT_PART(FORMATTEDADDRESS, ',', 3)
             WHEN FORMATTEDADDRESS = 'BYOB (No Corkage Fee), 60 Min Per Session, 104 W Washington Pl, New York, NY 10014, USA' 
                THEN SPLIT_PART(FORMATTEDADDRESS, ',', 4)
             WHEN FORMATTEDADDRESS = 'ALCOHOL & BYOB, 90 Min Per Session Please Be On Time, White Store, 464 Bergen St Front, Brooklyn, NY 11217, USA'
                THEN SPLIT_PART(FORMATTEDADDRESS, ',', 5)
             ELSE SPLIT_PART(FORMATTEDADDRESS, ',', 2) END AS CITY,
        -- Extract State and ZIP
    REVERSE(SPLIT_PART(TRIM(SPLIT_PART(REVERSE(FORMATTEDADDRESS), ',', 2)), ' ', 2)) AS STATE,
    REVERSE(SPLIT_PART(TRIM(SPLIT_PART(REVERSE(FORMATTEDADDRESS), ',', 2)), ' ', 1)) AS ZIP_CODE,
    CASE WHEN lower(TRIM(CITY)) IN ('new rochelle', 'bronx', 'pelham')
            THEN 'Bronx'
         WHEN lower(TRIM(CITY)) = 'new york'
            THEN 'Manhattan'
         WHEN lower(TRIM(CITY)) = 'brooklyn'
            THEN TRIM(CITY)
         WHEN lower(TRIM(CITY)) = 'staten island'
            THEN TRIM(CITY)
    ELSE 'Queens' END AS BOROUGH,
   // REGEXP_SUBSTR(FORMATTEDADDRESS, '^\\d+\\s?(-?)\\d* \\w+\\s*\\w* (\\w+\\s*\\d*)?{1,4}|[^Min Per Session]\\d+ \\w+ \\d*\\s?(\\w+\\s*\\d*)?{1,4}|\\d+\\w* \\w+ \\w+|(^\\d+ \\w+)|^\\S \\d+\\w+ \\w+', 1,1,'i') AS test_address
    
FROM {{ source('raw_data_sources', 'RAW_SUSHI_PLACES') }}
)

SELECT *
FROM STD_ADDRESSES