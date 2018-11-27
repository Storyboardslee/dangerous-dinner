from os.path import join
DATA_DIR = 'data'
FILES_TO_URLS = \
{
	'raw/human_es_to_endoderm_cells.txt':
		'https://obj.umiacs.umd.edu/xcl/CMSC828O-dangerous-dinner/stamlab_for_data3.txt',
	'raw/mouse_embryo_fibro_to_myocytes.txt':
		'https://obj.umiacs.umd.edu/xcl/CMSC828O-dangerous-dinner/stamlab_for_data2.txt',
}

DATA_TEMPLATE = join(DATA_DIR, '{filename}')
DATA_FILES = expand(DATA_TEMPLATE, filename=FILES_TO_URLS.keys())
OUTPUT_DIR = join(DATA_DIR, 'processed')

ADJACENCY_MATRIX = join(OUTPUT_DIR,'{organism}_adj_m.txt')
RAW_INPUT = join(DATA_DIR,'raw/{organism}.txt')

rule download:
	input:
		DATA_FILES

rule download_one:
    output:
        DATA_TEMPLATE
    params:
        url = lambda w: FILES_TO_URLS[w['filename']]
    shell:
        'wget -O {output} {params.url}'

rule adjacency_matrix:
	input:
		RAW_INPUT
	output:
		ADJACENCY_MATRIX
	shell:
		'''
		python scripts/gen_adjacency_matrix.py \
		-i {input} \
		-o {output}
		'''
