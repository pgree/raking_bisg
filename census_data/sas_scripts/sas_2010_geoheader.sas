/* this file comes from: https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.2010.html
/* This SAS program will make a SAS DATASET from the GEO Header Segment */
/* User will have to modify the libname,data and infile statements to conform to his environment*/
/* */
libname xxx 'the location of the files on your computer';
data xxx.mheader;
infile 'the location of the files on your computer\msgeo2010.pl' lrecl = 500 missover pad;
INPUT
@1 FILEID $6. /*File Identification*/
@7 STUSAB $2. /*State/US-Abbreviation (USPS)*/
@9 SUMLEV $3. /*Summary Level*/
@12 GEOCOMP $2. /*Geographic Component*/
@14 CHARITER $3. /*Characteristic Iteration*/
@17 CIFSN $2. /*Characteristic Iteration File Sequence Number*/
@19 LOGRECNO $7. /*Logical Record Number*/
@26 REGION $1. /*Region*/
@27 DIVISION $1. /*Division*/
@28 STATE $2. /*State (FIPS)*/
@30 COUNTY $3. /*County*/
@33 COUNTYCC $2. /*FIPS County Class Code*/
@35 COUNTYSC $2. /*County Size Code*/
@37 COUSUB $5. /*County Subdivision (FIPS)*/
@42 COUSUBCC $2. /*FIPS County Subdivision Class Code*/
@44 COUSUBSC $2. /*County Subdivision Size Code*/
@46 PLACE $5. /*Place (FIPS)*/
@51 PLACECC $2. /*FIPS Place Class Code*/
@53 PLACESC $2. /*Place Size Code*/
@55 TRACT $6. /*Census Tract*/
@61 BLKGRP $1. /*Block Group*/
@62 BLOCK $4. /*Block*/
@66 IUC $2. /*Internal Use Code*/
@68 CONCIT $5. /*Consolidated City (FIPS)*/
@73 CONCITCC $2. /*FIPS Consolidated City Class Code*/
@75 CONCITSC $2. /*Consolidated City Size Code*/
@77 AIANHH $4. /*American Indian Area/Alaska Native Area/Hawaiian Home Land (Census)*/
@81 AIANHHFP $5. /*American Indian Area/Alaska Native Area/Hawaiian Home Land (FIPS)*/
@86 AIANHHCC $2. /*FIPS American Indian Area/Alaska Native Area/Hawaiian Home Land Class Code*/
@88 AIHHTLI $1. /*American Indian Trust Land/Hawaiian Home Land Indicator*/
@89 AITSCE $3. /*American Indian Tribal Subdivision (Census)*/
@92 AITS $5. /*American Indian Tribal Subdivision (FIPS)*/
@97 AITSCC $2. /*FIPS American Indian Tribal Subdivision Class Code*/
@99 TTRACT $6. /*Tribal Census Tract*/
@105 TBLKGRP $1. /*Tribal Block Group*/
@106 ANRC $5. /*Alaska Native Regional Corporation (FIPS)*/
@111 ANRCCC $2. /*FIPS Alaska Native Regional Corporation Class Code*/
@113 CBSA $5. /*Metropolitan Statistical Area/Micropolitan Statistical Area*/
@118 CBSASC $2. /*Metropolitan Statistical Area/Micropolitan Statistical Area Size Code*/
@120 METDIV $5. /*Metropolitan Division*/
@125 CSA $3. /*Combined Statistical Area*/
@128 NECTA $5. /*New England City and Town Area*/
@133 NECTASC $2. /*New England City and Town Area Size Code*/
@135 NECTADIV $5. /*New England City and Town Area Division*/
@140 CNECTA $3. /*Combined New England City and Town Area*/
@143 CBSAPCI $1. /*Metropolitan Statistical Area/Micropolitan Statistical Area Principal City Indicator*/
@144 NECTAPCI $1. /*New England City and Town Area Principal City Indicator*/
@145 UA $5. /*Urban Area*/
@150 UASC $2. /*Urban Area Size Code*/
@152 UATYPE $1. /*Urban Area Type*/
@153 UR $1. /*Urban/Rural*/
@154 CD $2. /*Congressional District (111th)*/
@156 SLDU $3. /*State Legislative District (Upper Chamber) (Year 1)*/
@159 SLDL $3. /*State Legislative District (Lower Chamber) (Year 1)*/
@162 VTD $6. /*Voting District*/
@168 VTDI $1. /*Voting District Indicator*/
@169 RESERVE2 $3. /*Reserved*/
@172 ZCTA5 $5. /*ZIP Code Tabulation Area (5 digit)*/
@177 SUBMCD $5. /*Subminor Civil Division (FIPS)*/
@182 SUBMCDCC $2. /*FIPS Subminor Civil Division Class Code*/
@184 SDELM $5. /*School District (Elementary)*/
@189 SDSEC $5. /*School District (Secondary)*/
@194 SDUNI $5. /*School District (Unified)*/
@199 AREALAND $14. /*Area (Land)*/
@213 AREAWATR $14. /*Area (Water)*/
@227 NAME $90. /*Area Name-Legal/Statistical Area Description (LSAD) Term-Part Indicator*/
@317 FUNCSTAT $1. /*Functional Status Code*/
@318 GCUNI $1. /*Geographic Change User Note Indicator*/
@319 POP100 $9. /*Population Count (100%)*/
@328 HU100 $9. /*Housing Unit Count (100%)*/
@337 INTPTLAT $11. /*Internal Point (Latitude)*/
@348 INTPTLON $12. /*Internal Point (Longitude)*/
@360 LSADC $2. /*Legal/Statistical Area Description Code*/
@362 PARTFLAG $1. /*Part Flag*/
@363 RESERVE3 $6. /*Reserved*/
@369 UGA $5. /*Urban Growth Area*/
@374 STATENS $8. /*State (ANSI)*/
@382 COUNTYNS $8. /*County (ANSI)*/
@390 COUSUBNS $8. /*County Subdivision (ANSI)*/
@398 PLACENS $8. /*Place (ANSI)*/
@406 CONCITNS $8. /*Consolidated City (ANSI)*/
@414 AIANHHNS $8. /*American Indian Area/Alaska Native Area/Hawaiian Home Land (ANSI)*/
@422 AITSNS $8. /*American Indian Tribal Subdivision (ANSI)*/
@430 ANRCNS $8. /*Alaska Native Regional Corporation (ANSI)*/
@438 SUBMCDNS $8. /*Subminor Civil Division (ANSI)*/
@446 CD113 $2. /*Congressional District (113th)*/
@448 CD114 $2. /*Congressional District (114th)*/
@450 CD115 $2. /*Congressional District (115th)*/
@452 SLDU2 $3. /*State Legislative District (Upper Chamber) (Year 2)*/
@455 SLDU3 $3. /*State Legislative District (Upper Chamber) (Year 3)*/
@458 SLDU4 $3. /*State Legislative District (Upper Chamber) (Year 4)*/
@461 SLDL2 $3. /*State Legislative District (Lower Chamber) (Year 2)*/
@464 SLDL3 $3. /*State Legislative District (Lower Chamber) (Year 3)*/
@467 SLDL4 $3. /*State Legislative District (Lower Chamber) (Year 4)*/
@470 AIANHHSC $2. /*American Indian Area/Alaska Native Area/Hawaiian Home Land Size Code*/
@472 CSASC $2. /*Combined Statistical Area Size Code*/
@474 CNECTASC $2. /*Combined NECTA Size Code*/
@476 MEMI $1. /*Metropolitan Micropolitan Indicator*/
@477 NMEMI $1. /*NECTA Metropolitan Micropolitan Indicator*/
@478 PUMA $5. /*Public Use Microdata Area*/
@483 RESERVED $18. /*Reserved*/;
run;
