/* This SAS program will make a SAS DATASET from data part 2 Segment */
/* User will have to modify the libname,data and infile statements to conform to his environment*/
/* */
libname xxx 'the location of the files on your computer';
data xxx.mpart2;
infile 'the location of the files on your computer\ms000022010.pl' lrecl=20000 dlm=',' dsd missover pad;
LENGTH FILEID $6 /*File Identification*/
STUSAB $2 /*State/US-Abbreviation (USPS)*/
CHARITER $3 /*Characteristic Iteration*/
CIFSN   $2 /*Characteristic Iteration File Sequence Number*/
LOGRECNO $7 /*Logical Record Number*/
P0030001 $9 /*P3-1: Total*/
P0030002 $9 /*P3-2: Population of one race*/
P0030003 $9 /*P3-3: White alone*/
P0030004 $9 /*P3-4: Black or African American alone*/
P0030005 $9 /*P3-5: American Indian and Alaska Native alone*/
P0030006 $9 /*P3-6: Asian alone*/
P0030007 $9 /*P3-7: Native Hawaiian and Other Pacific Islander alone*/
P0030008 $9 /*P3-8: Some other race alone*/
P0030009 $9 /*P3-9: Population of two or more races*/
P0030010 $9 /*P3-10: Population of two races*/
P0030011 $9 /*P3-11: White; Black or African American*/
P0030012 $9 /*P3-12: White; American Indian and Alaska Native*/
P0030013 $9 /*P3-13: White; Asian*/
P0030014 $9 /*P3-14: White; Native Hawaiian and Other Pacific Islander*/
P0030015 $9 /*P3-15: White; Some other race*/
P0030016 $9 /*P3-16: Black or African American; American Indian and Alaska Native*/
P0030017 $9 /*P3-17: Black or African American; Asian*/
P0030018 $9 /*P3-18: Black or African American; Native Hawaiian and Other Pacific Islander*/
P0030019 $9 /*P3-19: Black or African American; Some other race*/
P0030020 $9 /*P3-20: American Indian and Alaska Native; Asian*/
P0030021 $9 /*P3-21: American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0030022 $9 /*P3-22: American Indian and Alaska Native; Some other race*/
P0030023 $9 /*P3-23: Asian; Native Hawaiian and Other Pacific Islander*/
P0030024 $9 /*P3-24: Asian; Some other race*/
P0030025 $9 /*P3-25: Native Hawaiian and Other Pacific Islander; Some other race*/
P0030026 $9 /*P3-26: Population of three races*/
P0030027 $9 /*P3-27: White; Black or African American; American Indian and Alaska Native*/
P0030028 $9 /*P3-28: White; Black or African American; Asian*/
P0030029 $9 /*P3-29: White; Black or African American; Native Hawaiian and Other Pacific Islander*/
P0030030 $9 /*P3-30: White; Black or African American; Some other race*/
P0030031 $9 /*P3-31: White; American Indian and Alaska Native; Asian*/
P0030032 $9 /*P3-32: White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0030033 $9 /*P3-33: White; American Indian and Alaska Native; Some other race*/
P0030034 $9 /*P3-34: White; Asian; Native Hawaiian and Other Pacific Islander*/
P0030035 $9 /*P3-35: White; Asian; Some other race*/
P0030036 $9 /*P3-36: White; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030037 $9 /*P3-37: Black or African American; American Indian and Alaska Native; Asian*/
P0030038 $9 /*P3-38: Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0030039 $9 /*P3-39: Black or African American; American Indian and Alaska Native; Some other race*/
P0030040 $9 /*P3-40: Black or African American; Asian; Native Hawaiian and Other Pacific Islander*/
P0030041 $9 /*P3-41: Black or African American; Asian; Some other race*/
P0030042 $9 /*P3-42: Black or African American; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030043 $9 /*P3-43: American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0030044 $9 /*P3-44: American Indian and Alaska Native; Asian; Some other race*/
P0030045 $9 /*P3-45: American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030046 $9 /*P3-46: Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030047 $9 /*P3-47: Population of four races*/
P0030048 $9 /*P3-48: White; Black or African American; American Indian and Alaska Native; Asian*/
P0030049 $9 /*P3-49: White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0030050 $9 /*P3-50: White; Black or African American; American Indian and Alaska Native; Some other race*/
P0030051 $9 /*P3-51: White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander*/
P0030052 $9 /*P3-52: White; Black or African American; Asian; Some other race*/
P0030053 $9 /*P3-53: White; Black or African American; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030054 $9 /*P3-54: White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0030055 $9 /*P3-55: White; American Indian and Alaska Native; Asian; Some other race*/
P0030056 $9 /*P3-56: White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030057 $9 /*P3-57: White; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030058 $9 /*P3-58: Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0030059 $9 /*P3-59: Black or African American; American Indian and Alaska Native; Asian; Some other race*/
P0030060 $9 /*P3-60: Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030061 $9 /*P3-61: Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030062 $9 /*P3-62: American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030063 $9 /*P3-63: Population of five races*/
P0030064 $9 /*P3-64: White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0030065 $9 /*P3-65: White; Black or African American; American Indian and Alaska Native; Asian; Some other race*/
P0030066 $9 /*P3-66: White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030067 $9 /*P3-67: White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030068 $9 /*P3-68: White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030069 $9 /*P3-69: Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0030070 $9 /*P3-70: Population of six races*/
P0030071 $9 /*P3-71: White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040001 $9 /*P4-1: Total*/
P0040002 $9 /*P4-2: Hispanic or Latino*/
P0040003 $9 /*P4-3: Not Hispanic or Latino*/
P0040004 $9 /*P4-4: Population of one race*/
P0040005 $9 /*P4-5: White alone*/
P0040006 $9 /*P4-6: Black or African American alone*/
P0040007 $9 /*P4-7: American Indian and Alaska Native alone*/
P0040008 $9 /*P4-8: Asian alone*/
P0040009 $9 /*P4-9: Native Hawaiian and Other Pacific Islander alone*/
P0040010 $9 /*P4-10: Some other race alone*/
P0040011 $9 /*P4-11: Population of two or more races*/
P0040012 $9 /*P4-12: Population of two races*/
P0040013 $9 /*P4-13: White; Black or African American*/
P0040014 $9 /*P4-14: White; American Indian and Alaska Native*/
P0040015 $9 /*P4-15: White; Asian*/
P0040016 $9 /*P4-16: White; Native Hawaiian and Other Pacific Islander*/
P0040017 $9 /*P4-17: White; Some other race*/
P0040018 $9 /*P4-18: Black or African American; American Indian and Alaska Native*/
P0040019 $9 /*P4-19: Black or African American; Asian*/
P0040020 $9 /*P4-20: Black or African American; Native Hawaiian and Other Pacific Islander*/
P0040021 $9 /*P4-21: Black or African American; Some other race*/
P0040022 $9 /*P4-22: American Indian and Alaska Native; Asian*/
P0040023 $9 /*P4-23: American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0040024 $9 /*P4-24: American Indian and Alaska Native; Some other race*/
P0040025 $9 /*P4-25: Asian; Native Hawaiian and Other Pacific Islander*/
P0040026 $9 /*P4-26: Asian; Some other race*/
P0040027 $9 /*P4-27: Native Hawaiian and Other Pacific Islander; Some other race*/
P0040028 $9 /*P4-28: Population of three races*/
P0040029 $9 /*P4-29: White; Black or African American; American Indian and Alaska Native*/
P0040030 $9 /*P4-30: White; Black or African American; Asian*/
P0040031 $9 /*P4-31: White; Black or African American; Native Hawaiian and Other Pacific Islander*/
P0040032 $9 /*P4-32: White; Black or African American; Some other race*/
P0040033 $9 /*P4-33: White; American Indian and Alaska Native; Asian*/
P0040034 $9 /*P4-34: White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0040035 $9 /*P4-35: White; American Indian and Alaska Native; Some other race*/
P0040036 $9 /*P4-36: White; Asian; Native Hawaiian and Other Pacific Islander*/
P0040037 $9 /*P4-37: White; Asian; Some other race*/
P0040038 $9 /*P4-38: White; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040039 $9 /*P4-39: Black or African American; American Indian and Alaska Native; Asian*/
P0040040 $9 /*P4-40: Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0040041 $9 /*P4-41: Black or African American; American Indian and Alaska Native; Some other race*/
P0040042 $9 /*P4-42: Black or African American; Asian; Native Hawaiian and Other Pacific Islander*/
P0040043 $9 /*P4-43: Black or African American; Asian; Some other race*/
P0040044 $9 /*P4-44: Black or African American; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040045 $9 /*P4-45: American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0040046 $9 /*P4-46: American Indian and Alaska Native; Asian; Some other race*/
P0040047 $9 /*P4-47: American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040048 $9 /*P4-48: Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040049 $9 /*P4-49: Population of four races*/
P0040050 $9 /*P4-50: White; Black or African American; American Indian and Alaska Native; Asian*/
P0040051 $9 /*P4-51: White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander*/
P0040052 $9 /*P4-52: White; Black or African American; American Indian and Alaska Native; Some other race*/
P0040053 $9 /*P4-53: White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander*/
P0040054 $9 /*P4-54: White; Black or African American; Asian; Some other race*/
P0040055 $9 /*P4-55: White; Black or African American; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040056 $9 /*P4-56: White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0040057 $9 /*P4-57: White; American Indian and Alaska Native; Asian; Some other race*/
P0040058 $9 /*P4-58: White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040059 $9 /*P4-59: White; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040060 $9 /*P4-60: Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0040061 $9 /*P4-61: Black or African American; American Indian and Alaska Native; Asian; Some other race*/
P0040062 $9 /*P4-62: Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040063 $9 /*P4-63: Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040064 $9 /*P4-64: American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040065 $9 /*P4-65: Population of five races*/
P0040066 $9 /*P4-66: White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander*/
P0040067 $9 /*P4-67: White; Black or African American; American Indian and Alaska Native; Asian; Some other race*/
P0040068 $9 /*P4-68: White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040069 $9 /*P4-69: White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040070 $9 /*P4-70: White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040071 $9 /*P4-71: Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
P0040072 $9 /*P4-72: Population of six races*/
P0040073 $9 /*P4-73: White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some other race*/
H0010001 $9 /*H1-1: Total*/
H0010002 $9 /*H1-2: Occupied*/
H0010003 $9 /*H1-3: Vacant*/ ;
INPUT
FILEID $
STUSAB $
CHARITER $
CIFSN $
LOGRECNO $
P0030001 $
P0030002 $
P0030003 $
P0030004 $
P0030005 $
P0030006 $
P0030007 $
P0030008 $
P0030009 $
P0030010 $
P0030011 $
P0030012 $
P0030013 $
P0030014 $
P0030015 $
P0030016 $
P0030017 $
P0030018 $
P0030019 $
P0030020 $
P0030021 $
P0030022 $
P0030023 $
P0030024 $
P0030025 $
P0030026 $
P0030027 $
P0030028 $
P0030029 $
P0030030 $
P0030031 $
P0030032 $
P0030033 $
P0030034 $
P0030035 $
P0030036 $
P0030037 $
P0030038 $
P0030039 $
P0030040 $
P0030041 $
P0030042 $
P0030043 $
P0030044 $
P0030045 $
P0030046 $
P0030047 $
P0030048 $
P0030049 $
P0030050 $
P0030051 $
P0030052 $
P0030053 $
P0030054 $
P0030055 $
P0030056 $
P0030057 $
P0030058 $
P0030059 $
P0030060 $
P0030061 $
P0030062 $
P0030063 $
P0030064 $
P0030065 $
P0030066 $
P0030067 $
P0030068 $
P0030069 $
P0030070 $
P0030071 $
P0040001 $
P0040002 $
P0040003 $
P0040004 $
P0040005 $
P0040006 $
P0040007 $
P0040008 $
P0040009 $
P0040010 $
P0040011 $
P0040012 $
P0040013 $
P0040014 $
P0040015 $
P0040016 $
P0040017 $
P0040018 $
P0040019 $
P0040020 $
P0040021 $
P0040022 $
P0040023 $
P0040024 $
P0040025 $
P0040026 $
P0040027 $
P0040028 $
P0040029 $
P0040030 $
P0040031 $
P0040032 $
P0040033 $
P0040034 $
P0040035 $
P0040036 $
P0040037 $
P0040038 $
P0040039 $
P0040040 $
P0040041 $
P0040042 $
P0040043 $
P0040044 $
P0040045 $
P0040046 $
P0040047 $
P0040048 $
P0040049 $
P0040050 $
P0040051 $
P0040052 $
P0040053 $
P0040054 $
P0040055 $
P0040056 $
P0040057 $
P0040058 $
P0040059 $
P0040060 $
P0040061 $
P0040062 $
P0040063 $
P0040064 $
P0040065 $
P0040066 $
P0040067 $
P0040068 $
P0040069 $
P0040070 $
P0040071 $
P0040072 $
P0040073 $
H0010001 $
H0010002 $
H0010003 $;
run;
