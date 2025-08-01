# Import required libraries
import pandas as pd  # For data manipulation and analysis
import time  # For measuring execution time
from collections import defaultdict, OrderedDict  # For efficient dictionary operations
from itertools import combinations  # For generating combinations of items
import os  # For operating system dependent functionality
import matplotlib.pyplot as plt  # For visualization
import networkx as nx  # For graph visualization
from networkx.drawing.nx_agraph import graphviz_layout  # For graph layout
import pydot  # For graph visualization

class FPTreeNode:
    def __init__(self, name, count, parent):
        """
        Node class for FP-Tree
        
        Args:
            name (str): Item name (None for root)
            count (int): Support count
            parent (FPTreeNode): Parent node reference
        """
        self.name = name  # Item name
        self.count = count  # Support count
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes dictionary
        self.link = None  # Link to next node with same item name

def visualize_fp_tree(root, header_table, item_initials, output_path):
    """
    Visualize the FP-Tree structure and save as image
    
    Args:
        root (FPTreeNode): Root of the FP-Tree
        header_table (dict): Header table of the FP-Tree
        item_initials (dict): Mapping of item names to initials
        output_path (str): Path to save the visualization image
    """
    
    # Create directed graph
    graph = pydot.Dot(graph_type='digraph', rankdir='TB')  # Top to bottom layout
    
    # Recursive function to add nodes and edges
    def add_nodes_edges(node, parent_id=None):
        """
        Recursively add nodes and edges to the graph
        
        Args:
            node (FPTreeNode): Current node to process
            parent_id (str): ID of parent node (None for root)
        """
        # Create node label
        if node.name is None:
            node_label = "Root"  # Root node label
        else:
            # Show item initial and count
            node_label = f"{item_initials.get(node.name, node.name)}\n({node.count})"
        
        # Create unique node ID
        node_id = str(id(node))
        # Set node color (lightblue for root, lightgreen for others)
        color = 'lightblue' if node.name is None else 'lightgreen'
        
        # Add node to graph
        graph.add_node(pydot.Node(node_id, label=node_label, style='filled', fillcolor=color))
        
        # Add edge from parent if not root
        if parent_id is not None:
            graph.add_edge(pydot.Edge(parent_id, node_id))
        
        # Recursively process children
        for child in node.children.values():
            add_nodes_edges(child, node_id)
    
    # Start building graph from root
    add_nodes_edges(root)
    
    # Save visualization as JPEG
    graph.write_jpg(output_path)
    
    return output_path

def fp_growth_algorithm(one_hot_file, min_support=0.3, min_confidence=0.6, output_dir='output'):
    """
    Implement FP-Growth algorithm for association rule mining
    
    Args:
        one_hot_file (str): Path to one-hot encoded Excel file
        min_support (float): Minimum support threshold (default: 0.3)
        min_confidence (float): Minimum confidence threshold (default: 0.6)
        output_dir (str): Directory to save output files (default: 'output')
    
    Returns:
        dict: Dictionary containing all results and metrics
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read one-hot encoded data
    df = pd.read_excel(one_hot_file, sheet_name='One-Hot')
    
    # Extract only item columns (exclude ID and Year columns)
    item_columns = [col for col in df.columns if col not in ['ID Transaksi', 'Tahun']]
    transactions = df[item_columns]
    total_transactions = len(transactions)
    
    # Start timer to measure execution time
    start_time = time.time()
    
    # Step 1: Calculate support for all items
    item_counts = transactions.sum().to_dict()  # Count occurrences of each item
    
    # Create Frequent 1-itemset table
    frequent_1_itemset = pd.DataFrame({
        'Item': list(item_counts.keys()),
        'Jumlah Transaksi': list(item_counts.values()),
        'Support': [count/total_transactions for count in item_counts.values()],
        'Keterangan': ['Lolos' if count/total_transactions >= min_support else 'Tidak Lolos' 
                      for count in item_counts.values()]
    })
    
    # Sort by transaction count (descending)
    frequent_1_itemset = frequent_1_itemset.sort_values(by='Jumlah Transaksi', ascending=False)
    
    # Filter only items that meet minimum support
    frequent_items = {item: count for item, count in item_counts.items() 
                     if count/total_transactions >= min_support}
    
    # Sort frequent items by support (descending)
    sorted_items = sorted(frequent_items.items(), key=lambda x: (-x[1], x[0]))
    
    # Create mapping of item names to initials for visualization
    item_initials = {}
    used_initials = set()  # Track used initials to avoid duplicates
    for item, count in sorted_items:
        # Generate initial from first letters of each word
        words = item.split()
        initial = ''.join([word[0].upper() for word in words])
        # Handle duplicate initials by adding number
        i = 1
        while initial in used_initials:
            initial = initial + str(i)
            i += 1
        used_initials.add(initial)
        item_initials[item] = initial
    
    # Create item table with initials (only frequent items)
    item_table = pd.DataFrame({
        'Item': [item for item, _ in sorted_items],
        'Inisial': [item_initials[item] for item, _ in sorted_items],
        'Jumlah Transaksi': [count for _, count in sorted_items],
        'Support': [count/total_transactions for _, count in sorted_items]
    })
    
    # Step 2: Process transactions - sort and filter items
    processed_transactions = []
    for _, row in transactions.iterrows():
        # Get frequent items in this transaction
        transaction_items = [item for item in item_columns if row[item] == 1 and item in frequent_items]
        # Sort items by support count (descending)
        transaction_items.sort(key=lambda x: (-frequent_items[x], x))
        # Convert to initials for visualization
        transaction_initials = [item_initials[item] for item in transaction_items]
        processed_transactions.append(transaction_initials)
    
    # Create processed transaction table
    transaction_table = pd.DataFrame({
        'ID Transaksi': df['ID Transaksi'],
        'Item (Transaksi terurut)': [', '.join(items) for items in processed_transactions]
    })
    
    # Step 3: Build FP-Tree
    header_table = defaultdict(list)  # Header table for node links
    root = FPTreeNode(None, None, None)  # Create root node
    
    # Function to update FP-Tree with a transaction
    def update_tree(items, node, header_table):
        """
        Recursively update FP-Tree with items from a transaction
        
        Args:
            items (list): Remaining items to process
            node (FPTreeNode): Current node in the tree
            header_table (dict): Header table for node links
        """
        if not items:
            return  # Base case
        
        first_item = items[0]
        # If item exists in children, increment count
        if first_item in node.children:
            child = node.children[first_item]
            child.count += 1
        else:
            # Create new node
            child = FPTreeNode(first_item, 1, node)
            node.children[first_item] = child
            # Update header table with link to this node
            if header_table[first_item]:
                last_node = header_table[first_item][-1]
                last_node.link = child
            header_table[first_item].append(child)
        
        # Recursively process remaining items
        update_tree(items[1:], child, header_table)
    
    # Build FP-Tree by processing all transactions
    for transaction in processed_transactions:
        update_tree(transaction, root, header_table)
    
    # Visualize FP-Tree and save image
    fp_tree_image_path = os.path.join(output_dir, 'fp_tree.jpg')
    visualize_fp_tree(root, header_table, item_initials, fp_tree_image_path)
    
    # Step 4: Mining FP-Tree - with detailed steps
    mining_steps = []  # Store mining steps for documentation
    
    def mine_tree(header_table, min_support_count, prefix, frequent_itemsets, level=0):
        """
        Recursively mine the FP-Tree to find frequent itemsets
        
        Args:
            header_table (dict): Current header table
            min_support_count (int): Minimum support count threshold
            prefix (set): Current prefix itemset
            frequent_itemsets (list): Store found frequent itemsets
            level (int): Current recursion depth
        """
        # Sort items by their order in the header table
        items = [item for item in header_table.keys()]
        items.sort(key=lambda x: (header_table[x][0].name))
        
        for item in items:
            # Create new itemset by adding current item to prefix
            new_prefix = prefix.copy()
            new_prefix.add(item)
            # Calculate support count for this itemset
            support_count = sum([node.count for node in header_table[item]])
            # Add to frequent itemsets
            frequent_itemsets.append((new_prefix, support_count))
            
            # Step 1: Find conditional pattern base
            conditional_patterns = []
            for node in header_table[item]:
                # Trace path back to root
                prefix_path = []
                parent = node.parent
                while parent.name is not None:
                    prefix_path.append(parent.name)
                    parent = parent.parent
                if prefix_path:
                    conditional_patterns.append((prefix_path, node.count))
            
            # Store conditional pattern base info
            mining_steps.append({
                'Tahap': 'Conditional Pattern Base',
                'Item': item,
                'Pattern Base': [f"{path} (count: {count})" for path, count in conditional_patterns],
                'Level': level
            })
            
            # Step 2: Build conditional FP-Tree
            conditional_header = defaultdict(list)
            conditional_root = FPTreeNode(None, None, None)
            
            for pattern, count in conditional_patterns:
                # Sort pattern by header table order
                pattern.sort(key=lambda x: (header_table[x][0].name if x in header_table else ''))
                # Multiply pattern by count (to handle counts)
                for _ in range(count):
                    update_tree(pattern, conditional_root, conditional_header)
            
            # Store conditional FP-tree info if not empty
            if conditional_header:
                mining_steps.append({
                    'Tahap': 'Conditional FP-Tree',
                    'Item': item,
                    'Tree Items': list(conditional_header.keys()),
                    'Level': level
                })
                
                # Step 3: Recursively mine conditional FP-Tree
                mine_tree(conditional_header, min_support_count, new_prefix, frequent_itemsets, level+1)
                
                # Store frequent itemset info
                mining_steps.append({
                    'Tahap': 'Frequent Itemset',
                    'Item': item,
                    'Itemset': new_prefix,
                    'Support Count': support_count,
                    'Level': level
                })
    
    # Mine the FP-Tree to find all frequent itemsets
    frequent_itemsets = []
    mine_tree(header_table, min_support * total_transactions, set(), frequent_itemsets)
    
    # Convert mining steps to DataFrame
    mining_steps_df = pd.DataFrame(mining_steps)
    
    # Step 5: Generate association rules
    rules = []
    for itemset, support_count in frequent_itemsets:
        itemset = list(itemset)
        # Only generate rules for itemsets with 2+ items
        if len(itemset) > 1:
            support = support_count / total_transactions
            
            # Generate all non-empty subsets
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = set(antecedent)
                    consequent = set(itemset) - antecedent
                    
                    # Find support for antecedent
                    antecedent_support = 0
                    for itemset2, count in frequent_itemsets:
                        if antecedent.issubset(set(itemset2)):
                            antecedent_support = count / total_transactions
                            break
                    
                    # Calculate confidence
                    confidence = support / antecedent_support if antecedent_support > 0 else 0
                    
                    # Only keep rules meeting minimum confidence
                    if confidence >= min_confidence:
                        # Find support for consequent
                        consequent_support = 0
                        for itemset2, count in frequent_itemsets:
                            if consequent.issubset(set(itemset2)):
                                consequent_support = count / total_transactions
                                break
                        
                        # Calculate lift
                        lift = confidence / consequent_support if consequent_support > 0 else 0
                        
                        # Determine correlation type
                        correlation = "Positif" if lift > 1 else "Negatif" if lift < 1 else "Netral"
                        
                        # Convert initials back to original item names
                        antecedent_items = [list(item_initials.keys())[list(item_initials.values()).index(a)] 
                                          for a in antecedent]
                        consequent_items = [list(item_initials.keys())[list(item_initials.values()).index(c)] 
                                          for c in consequent]
                        
                        # Add rule to results
                        rules.append({
                            'Rule': f"{antecedent_items} => {consequent_items}",
                            'Support': round(support, 4),
                            'Support A': round(antecedent_support, 4),
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
        'frequent_1_itemset': frequent_1_itemset,
        'item_table': item_table,
        'transaction_table': transaction_table,
        'fp_tree_image': fp_tree_image_path,
        'mining_steps': mining_steps_df,
        'association_rules': pd.DataFrame(rules),
        'metrics': {
            'Waktu Eksekusi (detik)': round(execution_time, 2),
            'Jumlah Rule Ditemukan': num_rules,
            'Rata-rata Lift': round(avg_lift, 4),
            'Rata-rata Akurasi': round(avg_accuracy, 4),
            'Min Support': min_support,
            'Min Confidence': min_confidence,
            'Total Transaksi': total_transactions
        },
        'item_initials': item_initials
    }

def save_results_to_excel(results, output_file):
    """
    Save FP-Growth results to Excel file with multiple sheets
    
    Args:
        results (dict): Results from fp_growth_algorithm function
        output_file (str): Path to output Excel file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create Excel writer object
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Sheet 1: Original One-Hot data
        one_hot_df = pd.read_excel(results['input_file'], sheet_name='One-Hot')
        one_hot_df.to_excel(writer, sheet_name='One-Hot', index=False)
        
        # Sheet 2: FP-Growth Method Results
        workbook = writer.book
        worksheet = workbook.add_worksheet('Metode FP-Growth')
        writer.sheets['Metode FP-Growth'] = worksheet
        row = 0  # Track current row position
        
        # Write Frequent 1-itemset table
        worksheet.write(row, 0, 'Frequent 1-itemset')
        row += 1
        results['frequent_1_itemset'].to_excel(writer, sheet_name='Metode FP-Growth', 
                                             startrow=row, startcol=0, index=False)
        row += len(results['frequent_1_itemset']) + 2  # Move down for next section
        
        # Write Item and Initials table (frequent items only)
        worksheet.write(row, 0, 'Item dan Inisial (Lolos)')
        row += 1
        results['item_table'].to_excel(writer, sheet_name='Metode FP-Growth', 
                                     startrow=row, startcol=0, index=False)
        row += len(results['item_table']) + 2
        
        # Write Processed Transactions table
        worksheet.write(row, 0, 'Transaksi yang Diproses')
        row += 1
        results['transaction_table'].to_excel(writer, sheet_name='Metode FP-Growth',
                                           startrow=row, startcol=0, index=False)
        row += len(results['transaction_table']) + 2
        
        # Add FP-Tree visualization image to Excel
        worksheet.write(row, 0, 'Visualisasi FP-Tree')
        row += 1
        worksheet.insert_image(row, 0, results['fp_tree_image'], {'x_scale': 0.5, 'y_scale': 0.5})
        row += 21  # Adjust row position based on image height
        
        # Add mining steps
        worksheet.write(row, 0, 'Tahapan Mining FP-Tree')
        row += 1
        results['mining_steps'].to_excel(writer, sheet_name='Metode FP-Growth',
                                       startrow=row, startcol=0, index=False)
        row += len(results['mining_steps']) + 2
        
        # Add Association Rules
        worksheet.write(row, 0, 'Association Rules')
        row += 1
        results['association_rules'].to_excel(writer, sheet_name='Metode FP-Growth',
                                            startrow=row, startcol=0, index=False)
        row += len(results['association_rules']) + 2
        
        # Add Metrics
        worksheet.write(row, 0, 'Metrics')
        row += 1
        df_metrics = pd.DataFrame.from_dict(results['metrics'], orient='index', columns=['Value'])
        df_metrics.to_excel(writer, sheet_name='Metode FP-Growth',
                          startrow=row, startcol=0, header=True)
        
        # Auto-adjust column widths for all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            
            # Get appropriate DataFrame for this sheet
            if sheet_name == 'One-Hot':
                df = one_hot_df
            elif sheet_name == 'Metode FP-Growth':
                # Combine all DataFrames written to this sheet
                df_list = [
                    results['frequent_1_itemset'],
                    results['item_table'],
                    results['transaction_table'],
                    results['mining_steps'],
                    results['association_rules'],
                    df_metrics
                ]
                df = pd.concat(df_list, axis=0, ignore_index=True)
            else:
                continue
            
            # Iterate through columns to determine maximum width
            for idx, col in enumerate(df.columns):
                # Find maximum length in column
                max_len = max(
                    df[col].astype(str).str.len().max(),  # Data length
                    len(str(col))  # Header length
                ) if not df.empty else len(str(col))
                
                # Set column width (with some padding)
                worksheet.set_column(idx, idx, max_len + 2)

if __name__ == "__main__":
    # Set input/output parameters
    input_file = 'data_temp/tahap3_one_hot_transaksi.xlsx'
    output_file = 'data_temp/tahap5_hasil_perhitungan.xlsx'
    output_dir = 'data_temp/fp_growth_output'
    min_support = 0.3  # Minimum support threshold (30%)
    min_confidence = 0.6  # Minimum confidence threshold (60%)
    
    # Run FP-Growth algorithm
    print("Memulai perhitungan FP-Growth...")
    results = fp_growth_algorithm(input_file, min_support, min_confidence, output_dir)
    results['input_file'] = input_file  # Store input file path in results
    
    # Save results to Excel
    save_results_to_excel(results, output_file)
    print(f"\nHasil perhitungan disimpan di: {output_file}")
    
    # Print summary metrics
    print("\nRingkasan Hasil:")
    for key, value in results['metrics'].items():
        print(f"{key}: {value}")
    
    print(f"\nGambar FP-Tree disimpan di: {results['fp_tree_image']}")