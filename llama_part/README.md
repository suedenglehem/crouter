scripts for serving models (ggufs) locally with llama-server

i have 2 boxes, one with 4080s+4060ti 16gb (i7) and another (tr4) with 2 3090 + 3080ti

so, with config_i7.yml there both fast & deep models have api access, 
i'd run qwen38 27b on i7 (script deep.sh)  and qwen3.5 9b on 3080ti of tr4 (script fast.sh). this config
is for dev mostly, cuz where would be also qwen3.8 27b running on 2 3090 (script qw_uncensored_mtp_q8_claude.sh)
and claude code would use that model. it'd debug router on i7 and 3080ti.

for testing config.yaml which is without api (sure, i don't need api key on llama servers 
if i'm running full local), i'd run both qw9claude.sh and qw_uncensored_mtp_q8_claude.sh on tr4
