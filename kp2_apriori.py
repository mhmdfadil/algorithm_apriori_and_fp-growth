# Import required libraries
import pandas as pd  # For data manipulation and analysis
import time  # For measuring execution time
from itertools import combinations  # For generating combinations of items
from collections import defaultdict  # For efficient dictionary operations
import os  # For operating system dependent functionality

def apriori_algorithm(one_hot_file, min_support=0.3, min_confidence=0.6):
    """
    Implement the Apriori algorithm for association rule mining
    
    Args:
        one_hot_file (str): Path to the one-hot encoded Excel file
        min_support (float): Minimum support threshold (default: 0.3)
        min_confidence (float): Minimum confidence threshold (default: 0.6)
    
    Returns:
        dict: Dictionary containing frequent itemsets, association rules and metrics
    """
    
    # Read the one-hot encoded data
    df = pd.read_excel(one_hot_file, sheet_name='One-Hot')
    
    # Extract only the item columns (exclude ID and Year columns)
    item_columns = [col for col in df.columns if col not in ['ID Transaksi', 'Tahun']]
    transactions = df[item_columns]
    total_transactions = len(transactions)
    
    # Start timer to measure execution time
    start_time = time.time()
    
    # Function to calculate support count and support ratio
    def calculate_support(itemset):
        """
        Calculate support count and support ratio for an itemset
        
        Args:
            itemset (str or tuple): Single item or combination of items
            
        Returns:
            tuple: (support_count, support_ratio)
        """
        if isinstance(itemset, str):
            # For single itemset
            support_count = transactions[itemset].sum()
        else:
            # For itemset combinations
            support_count = transactions[list(itemset)].all(axis=1).sum()
        return support_count, support_count / total_transactions
    
    # Step 1: Find frequent 1-itemsets
    frequent_itemsets = {}  # Dictionary to store frequent itemsets by size
    k = 1  # Starting with 1-itemsets
    frequent_itemsets[k] = {}  # Initialize dictionary for 1-itemsets
    
    # Store detailed results for each frequent itemset level
    frequent_results = {k: []}
    
    # Calculate support for each individual item
    for item in item_columns:
        support_count, support = calculate_support(item)
        status = "Lolos" if support >= min_support else "Tidak Lolos"
        frequent_results[k].append({
            'Itemset': item,
            'Jumlah Transaksi': support_count,
            'Support': support,
            'Keterangan': status
        })
        # Store only itemsets that meet minimum support
        if support >= min_support:
            frequent_itemsets[k][frozenset([item])] = support
    
    print(f"Frequent {k}-itemset ditemukan: {len(frequent_itemsets[k])}")
    
    # Step 2: Iterate to find larger itemsets
    k = 2  # Move to 2-itemsets
    while True:
        frequent_itemsets[k] = {}  # Initialize dictionary for k-itemsets
        frequent_results[k] = []  # Initialize results storage for k-itemsets
        
        # Generate candidate itemsets from previous frequent itemsets
        previous_items = [item for itemset in frequent_itemsets[k-1].keys() for item in itemset]
        unique_previous_items = list(set(previous_items))  # Get unique items
        candidates = list(combinations(unique_previous_items, k))  # Generate combinations
        
        # Check each candidate itemset
        for candidate in candidates:
            # Prune step: Check if all subsets are frequent
            all_subsets_frequent = True
            for subset in combinations(candidate, k-1):
                if frozenset(subset) not in frequent_itemsets[k-1]:
                    all_subsets_frequent = False
                    break
            
            # Only proceed if all subsets are frequent
            if all_subsets_frequent:
                support_count, support = calculate_support(candidate)
                status = "Lolos" if support >= min_support else "Tidak Lolos"
                frequent_results[k].append({
                    'Itemset': ', '.join(candidate),
                    'Jumlah Transaksi': support_count,
                    'Support': support,
                    'Keterangan': status
                })
                # Store only itemsets that meet minimum support
                if support >= min_support:
                    frequent_itemsets[k][frozenset(candidate)] = support
        
        # Stop if no frequent itemsets found at this level
        if not frequent_itemsets[k]:
            del frequent_itemsets[k]  # Remove empty entry
            del frequent_results[k]  # Remove empty entry
            break
        
        print(f"Frequent {k}-itemset ditemukan: {len(frequent_itemsets[k])}")
        k += 1  # Move to next itemset size
    
    # Step 3: Generate association rules from frequent itemsets
    rules = []  # Store all valid association rules
    
    # Process itemsets of size 2 or larger
    for k, itemsets in list(frequent_itemsets.items())[1:]:
        for itemset in itemsets.keys():
            # Generate all possible non-empty subsets
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # Get support values
                    support_A = frequent_itemsets[len(antecedent)][antecedent]
                    support_B = frequent_itemsets[len(consequent)][consequent]
                    support_AUB = frequent_itemsets[len(itemset)][itemset]
                    
                    # Calculate confidence
                    confidence = support_AUB / support_A
                    
                    # Only keep rules meeting minimum confidence
                    if confidence >= min_confidence:
                        # Calculate lift
                        lift = confidence / support_B
                        
                        # Determine correlation type
                        correlation = "Positif" if lift > 1 else "Negatif" if lift < 1 else "Netral"
                        
                        # Add rule to results
                        rules.append({
                            'Rule': f"{set(antecedent)} => {set(consequent)}",
                            'Support A': round(support_A, 4),
                            'Support B': round(support_B, 4),
                            'Support AUB': round(support_AUB, 4),
                            'Confidence': round(confidence, 4),
                            'Lift': round(lift, 4),
                            'Korelasi': correlation
                        })
    
    # Calculate total execution time
    execution_time = time.time() - start_time
    
    # Calculate evaluation metrics
    num_rules = len(rules)
    avg_lift = sum(rule['Lift'] for rule in rules) / num_rules if num_rules > 0 else 0
    
    # Calculate additional metrics for each rule
    for rule in rules:
        precision = rule['Confidence']
        recall = precision  # In association rules, confidence is equivalent to recall
        rule['Akurasi'] = round((precision + recall) / 2, 4)
        rule['Recall'] = round(recall, 4)
        rule['Presisi'] = round(precision, 4)
        rule['F1-Score'] = round(2 * (precision * recall) / (precision + recall), 4) if (precision + recall) > 0 else 0

    # Calculate average accuracy across all rules
    avg_accuracy = sum(rule['Akurasi'] for rule in rules) / num_rules if num_rules > 0 else 0
    
    # Return all results in structured format
    return {
        'frequent_results': frequent_results,
        'association_rules': rules,
        'metrics': {
            'Waktu Eksekusi (detik)': round(execution_time, 2),
            'Jumlah Rule Ditemukan': num_rules,
            'Rata-rata Lift': round(avg_lift, 4),
            'Rata-rata Akurasi': round(avg_accuracy, 4),
            'Min Support': min_support,
            'Min Confidence': min_confidence,
            'Total Transaksi': total_transactions
        },
        'input_file': one_hot_file
    }

def save_results_to_excel(results, output_file):
    """
    Save Apriori algorithm results to Excel file
    
    Args:
        results (dict): Results from apriori_algorithm function
        output_file (str): Path to output Excel file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create Excel writer object
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Sheet 1: Original One-Hot data
        one_hot_df = pd.read_excel(results['input_file'], sheet_name='One-Hot')
        one_hot_df.to_excel(writer, sheet_name='One-Hot', index=False)
        
        # Sheet 2: Apriori Method Results
        workbook = writer.book
        worksheet = workbook.add_worksheet('Metode Apriori')
        writer.sheets['Metode Apriori'] = worksheet
        row = 0  # Track current row position
        
        # Write frequent itemsets by level
        for k in sorted(results['frequent_results'].keys()):
            worksheet.write(row, 0, f'Frequent {k}-Itemset')
            row += 1
            
            # Convert to DataFrame and write to Excel
            df_frequent = pd.DataFrame(results['frequent_results'][k])
            df_frequent.to_excel(writer, sheet_name='Metode Apriori', 
                               startrow=row, startcol=0, index=False)
            row += len(df_frequent) + 2  # Move down for next section
        
        # Write association rules
        worksheet.write(row, 0, 'Association Rules')
        row += 1
        df_rules = pd.DataFrame(results['association_rules'])
        df_rules.to_excel(writer, sheet_name='Metode Apriori',
                         startrow=row, startcol=0, index=False)
        row += len(df_rules) + 2
        
        # Write evaluation metrics
        worksheet.write(row, 0, 'Metrics')
        row += 1
        df_metrics = pd.DataFrame.from_dict(results['metrics'], orient='index', columns=['Value'])
        df_metrics.to_excel(writer, sheet_name='Metode Apriori',
                          startrow=row, startcol=0, header=True)
        
        # Auto-adjust column widths for all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            
            # Get appropriate DataFrame for this sheet
            if sheet_name == 'One-Hot':
                df = one_hot_df
            elif sheet_name == 'Metode Apriori':
                # Combine all DataFrames written to this sheet
                df_list = []
                for k in sorted(results['frequent_results'].keys()):
                    df_list.append(pd.DataFrame(results['frequent_results'][k]))
                df_list.append(pd.DataFrame(results['association_rules']))
                df_list.append(pd.DataFrame.from_dict(results['metrics'], orient='index', columns=['Value']))
                df = pd.concat(df_list, axis=0)
            else:
                continue
            
            # Iterate through columns to determine maximum width
            for idx, col in enumerate(df.columns):
                # Find maximum length in column
                max_len = max(
                    df[col].astype(str).str.len().max(),  # Data length
                    len(str(col))  # Header length
                )
                # Set column width (with some padding)
                worksheet.set_column(idx, idx, max_len + 2)

if __name__ == "__main__":
    # Set input/output parameters
    input_file = 'data_temp/tahap3_one_hot_transaksi.xlsx'
    output_file = 'data_temp/tahap4_hasil_perhitungan.xlsx'
    min_support = 0.3  # Minimum support threshold (30%)
    min_confidence = 0.6  # Minimum confidence threshold (60%)
    
    # Run Apriori algorithm
    print("Memulai perhitungan Apriori...")
    results = apriori_algorithm(input_file, min_support, min_confidence)
    
    # Save results to Excel
    save_results_to_excel(results, output_file)
    print(f"\nHasil perhitungan disimpan di: {output_file}")
    
    # Print summary metrics
    print("\nRingkasan Hasil:")
    for key, value in results['metrics'].items():
        print(f"{key}: {value}")