###############################################################################
# SNP–ATAC-seq Integration Pipeline
# Description: Intersects GWAS SNPs with ATAC-seq peaks and computes features
# Author: Mo Dehestani
# Date: July 2025
###############################################################################

# ------------------------------------------------------------------------------
# Load Required Libraries
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(GenomicRanges)
  library(EnsDb.Hsapiens.v86)
  library(Seurat)
  library(Matrix)
  library(matrixStats)
  library(parallel)
})

# ------------------------------------------------------------------------------
# Function to Validate Input File
# ------------------------------------------------------------------------------

validate_file <- function(file_path) {
  if (!file.exists(file_path)) {
    stop("Error: Input file '", file_path, "' does not exist.")
  }
  return(TRUE)
}

# ------------------------------------------------------------------------------
# Load GWAS Data
# ------------------------------------------------------------------------------

cat("Loading GWAS data...\n")
gwas_file <- "/data/nallsEtAl2019_allSamples_allVariants_withRSID_GRCh37.tab"
validate_file(gwas_file)
gwas_snps <- fread(gwas_file, nThread = parallel::detectCores() - 1)

gwas <- gwas_snps[, .(SNP, CHR = paste0("chr", CHR), BP, P = p, BETA = b)][
  P < 0.05 & !is.na(BP) & !is.na(CHR) & !is.na(SNP)
]

cat("Total SNPs after filtering (p < 0.05):", nrow(gwas), "\n")

# Create GRanges object
cat("Creating genomic ranges for SNPs...\n")
gwas_gr <- makeGRangesFromDataFrame(
  gwas,
  seqnames.field = "CHR",
  start.field = "BP",
  end.field = "BP",
  keep.extra.columns = TRUE
)
mcols(gwas_gr) <- gwas_gr@elementMetadata[, c("SNP", "P", "BETA")]
names(mcols(gwas_gr)) <- c("snp_id", "pvalue", "beta")

# ------------------------------------------------------------------------------
# Load and Subset ATAC-seq Data
# ------------------------------------------------------------------------------

cat("Loading ATAC-seq data...\n")
dan_atac_file <- "Foundin1_ATAC_DAN_iPD_HC.rds"
validate_file(dan_atac_file)
dan_atac <- readRDS(dan_atac_file)

if (!"mutationV2" %in% colnames(dan_atac@meta.data)) {
  stop("Error: 'mutationV2' column not found in ATAC-seq metadata.")
}
table(dan_atac@meta.data$mutationV2)

Idents(dan_atac) <- "mutationV2"
DAN <- subset(dan_atac, idents = c("iPD", "HC"))

# ------------------------------------------------------------------------------
# Step 1: Filter ATAC Peaks
# ------------------------------------------------------------------------------

cat("Filtering ATAC peaks...\n")
peaks_gr <- granges(DAN)
peaks_gr <- keepStandardChromosomes(peaks_gr, pruning.mode = "coarse")
peaks_gr <- resize(peaks_gr, width = 1, fix = "center")

# ------------------------------------------------------------------------------
# Step 2: Overlap SNPs with Peaks
# ------------------------------------------------------------------------------

cat("Finding overlaps between SNPs and peaks...\n")
snp_in_peaks <- subsetByOverlaps(gwas_gr, peaks_gr, ignore.strand = TRUE)
if (length(snp_in_peaks) == 0) warning("No overlaps found between GWAS SNPs and ATAC peaks.")
snp_peak_hits <- findOverlaps(snp_in_peaks, peaks_gr, ignore.strand = TRUE)

if (is.null(names(peaks_gr))) {
  peak_names <- paste0(seqnames(peaks_gr), "-", start(peaks_gr), "-", end(peaks_gr))
  names(peaks_gr) <- peak_names
  cat("Created peak names from coordinates\n")
} else {
  cat("Using existing peak names\n")
}

snp_peak_df <- data.frame(
  snp_idx = queryHits(snp_peak_hits),
  peak_idx = subjectHits(snp_peak_hits),
  stringsAsFactors = FALSE
)
snp_peak_df$rsID <- mcols(snp_in_peaks)$snp_id[snp_peak_df$snp_idx]
snp_peak_df$peak <- names(peaks_gr)[snp_peak_df$peak_idx]

cat("Data types in snp_peak_df:\n")
cat("snp_idx:", class(snp_peak_df$snp_idx), "\n")
cat("peak_idx:", class(snp_peak_df$peak_idx), "\n")
cat("rsID:", class(snp_peak_df$rsID), "\n")
cat("peak:", class(snp_peak_df$peak), "\n")
cat("Number of SNP-peak overlaps found:", nrow(snp_peak_df), "\n")
print(head(snp_peak_df))

# ------------------------------------------------------------------------------
# Step 3: Create Accessibility Matrix
# ------------------------------------------------------------------------------

cat("Computing accessibility metrics...\n")
accessibility <- AverageExpression(DAN, group.by = "mutationV2", assays = "ATAC")$ATAC
peak_acc_all <- rowMeans(GetAssayData(DAN, assay = "ATAC", slot = "data"))

# ------------------------------------------------------------------------------
# Step 4: Compute Peak Features
# ------------------------------------------------------------------------------

cat("Computing peak features...\n")
peak_features <- data.frame(
  peak = names(peak_acc_all),
  acc_mean = peak_acc_all,
  acc_PD = accessibility[, "iPD"],
  acc_HC = accessibility[, "HC"],
  acc_diff = accessibility[, "iPD"] - accessibility[, "HC"]
)

threshold <- 0.1
binary_access <- GetAssayData(DAN, slot = "data", assay = "ATAC") > threshold
cell_specificity <- Matrix::rowSums(binary_access) / ncol(binary_access)
peak_features$cell_specificity <- cell_specificity

cat("Calculating peak widths directly from peak names...\n")
calculate_width_from_name <- function(peak_name) {
  parts <- strsplit(peak_name, "-")[[1]]
  if (length(parts) == 3) {
    start_pos <- as.numeric(parts[2])
    end_pos <- as.numeric(parts[3])
    if (!is.na(start_pos) && !is.na(end_pos)) {
      return(end_pos - start_pos + 1)
    }
  }
  return(NA)
}
peak_features$width <- sapply(peak_features$peak, calculate_width_from_name)
cat("Number of NA values in width:", sum(is.na(peak_features$width)), "\n")
print(summary(peak_features$width))
print(head(peak_features))

# ------------------------------------------------------------------------------
# Step 5: TSS Distance Calculation
# ------------------------------------------------------------------------------

cat("Calculating TSS distances with chromosome naming fix...\n")
peak_names <- peak_features$peak
peak_coords <- strsplit(peak_names, "-")
peak_df <- data.frame(
  chr = sapply(peak_coords, function(x) x[1]),
  start = as.numeric(sapply(peak_coords, function(x) x[2])),
  end = as.numeric(sapply(peak_coords, function(x) x[3])),
  peak_name = peak_names,
  stringsAsFactors = FALSE
)
peak_df <- peak_df[complete.cases(peak_df), ]
cat("Successfully parsed", nrow(peak_df), "peaks\n")

all_peaks_gr <- makeGRangesFromDataFrame(
  peak_df,
  seqnames.field = "chr",
  start.field = "start",
  end.field = "end",
  keep.extra.columns = TRUE
)
names(all_peaks_gr) <- peak_df$peak_name

tss_annot <- genes(EnsDb.Hsapiens.v86)
tss_annot <- resize(tss_annot, width = 1, fix = "start")
tss_annot <- keepStandardChromosomes(tss_annot, pruning.mode = "coarse")
seqlevels(tss_annot) <- paste0("chr", seqlevels(tss_annot))

all_peaks_gr <- keepStandardChromosomes(all_peaks_gr, pruning.mode = "coarse")
tss_annot <- keepStandardChromosomes(tss_annot, pruning.mode = "coarse")

nearest_tss <- distanceToNearest(all_peaks_gr, tss_annot, ignore.strand = TRUE)
tss_distances <- rep(NA, length(all_peaks_gr))
if (length(nearest_tss) > 0) {
  tss_distances[queryHits(nearest_tss)] <- mcols(nearest_tss)$distance
}
peak_features$tss_distance <- tss_distances[match(peak_features$peak, names(all_peaks_gr))]

cat("TSS distance summary:\n")
print(summary(peak_features$tss_distance))
print(head(peak_features))

# ------------------------------------------------------------------------------
# Step 6: Merge SNP and Peak Features (Full-width Peaks)
# ------------------------------------------------------------------------------

cat("Getting original ATAC peaks...\n")
original_peaks_gr <- granges(DAN)
original_peaks_gr <- keepStandardChromosomes(original_peaks_gr, pruning.mode = "coarse")
original_peak_names <- rownames(DAN@assays$ATAC@data)
names(original_peaks_gr) <- original_peak_names

cat("Finding overlaps between SNPs and ORIGINAL peaks...\n")
snp_in_peaks <- subsetByOverlaps(gwas_gr, original_peaks_gr, ignore.strand = TRUE)
snp_peak_hits <- findOverlaps(snp_in_peaks, original_peaks_gr, ignore.strand = TRUE)

snp_peak_df <- data.frame(
  snp_idx = queryHits(snp_peak_hits),
  peak_idx = subjectHits(snp_peak_hits),
  stringsAsFactors = FALSE
)
snp_peak_df$rsID <- mcols(snp_in_peaks)$snp_id[snp_peak_df$snp_idx]
snp_peak_df$peak <- names(original_peaks_gr)[snp_peak_df$peak_idx]

cat("Checking if names match peak_features:\n")
name_matches <- intersect(snp_peak_df$peak, peak_features$peak)
cat("Number of matching names:", length(name_matches), "\n")

cat("Attempting merge...\n")
merged <- left_join(snp_peak_df, peak_features, by = "peak")
cat("Merge results:\n")
cat("Total rows:", nrow(merged), "\n")
cat("Rows with accessibility data:", sum(!is.na(merged$acc_mean)), "\n")
cat("Rows with NA accessibility:", sum(is.na(merged$acc_mean)), "\n")
print(head(merged))

# ------------------------------------------------------------------------------
# Step 7: Add GWAS Data (Next Step Placeholder)
# ------------------------------------------------------------------------------

# Add GWAS data to the merged table (TO BE IMPLEMENTED)
