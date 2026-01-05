#!/usr/bin/env python3
"""
Complete indentation fix for unified_uel_ai_system.py
This script will fix all indentation issues in your file
"""

import re
import sys

def fix_entire_file_indentation(input_file, output_file):
    """Fix all indentation issues in the Python file"""
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    
    # Track state
    current_class = None
    current_method = None
    in_class = False
    in_method = False
    in_docstring = False
    method_body_indent = 8  # Standard for method body in a class
    
    # Track nested structures
    indent_stack = [0]  # Track indentation levels
    
    for i, line in enumerate(lines):
        original_line = line
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            continue
        
        # Skip comments at module level
        if stripped.startswith('#') and not in_class:
            fixed_lines.append(line)
            continue
        
        # Handle module-level functions (like get_logger)
        if stripped.startswith('def ') and not in_class:
            in_method = True
            current_method = stripped
            fixed_lines.append(stripped + '\n')
            continue
        
        # Handle module-level return (for functions outside classes)
        if stripped.startswith('return') and not in_class and in_method:
            fixed_lines.append('    ' + stripped + '\n')
            if not lines[i+1].strip() or lines[i+1].strip().startswith('def ') or lines[i+1].strip().startswith('class '):
                in_method = False
            continue
        
        # Class definition
        if stripped.startswith('class '):
            in_class = True
            in_method = False
            current_class = stripped
            fixed_lines.append(stripped + '\n')
            print(f"Processing class: {stripped[:50]}")
            continue
        
        # Check if we're leaving a class (next class or module-level code)
        if in_class and not in_method:
            # If we see another class or a module-level function/variable
            if (stripped.startswith('class ') and current_class not in stripped) or \
               (stripped.startswith('def ') and i > 0 and not lines[i-1].strip().startswith('@')) or \
               (stripped.startswith(('config =', 'research_config =', '__all__'))):
                in_class = False
                current_class = None
                fixed_lines.append(stripped + '\n')
                continue
        
        # Method definition in a class
        if in_class and stripped.startswith('def '):
            in_method = True
            current_method = stripped
            fixed_lines.append('    ' + stripped + '\n')  # 4 spaces for methods in class
            continue
        
        # Handle decorators
        if in_class and stripped.startswith('@'):
            fixed_lines.append('    ' + stripped + '\n')  # 4 spaces for decorators in class
            continue
        
        # Docstrings
        if '"""' in stripped or "'''" in stripped:
            if in_method and in_class:
                # Docstring inside a method in a class
                fixed_lines.append('        ' + stripped + '\n')  # 8 spaces
                in_docstring = not in_docstring if stripped.count('"""') == 1 else False
            elif in_class and not in_method:
                # Class-level docstring
                fixed_lines.append('    ' + stripped + '\n')  # 4 spaces
                in_docstring = not in_docstring if stripped.count('"""') == 1 else False
            elif in_method and not in_class:
                # Module-level function docstring
                fixed_lines.append('    ' + stripped + '\n')  # 4 spaces
                in_docstring = not in_docstring if stripped.count('"""') == 1 else False
            else:
                fixed_lines.append(stripped + '\n')
            continue
        
        # Content inside docstring
        if in_docstring:
            if in_method and in_class:
                fixed_lines.append('        ' + stripped + '\n')  # 8 spaces
            elif in_class and not in_method:
                fixed_lines.append('    ' + stripped + '\n')  # 4 spaces
            else:
                fixed_lines.append('    ' + stripped + '\n')  # 4 spaces
            continue
        
        # Method body content
        if in_method and in_class:
            # Calculate proper indentation based on content
            if stripped.startswith('return'):
                # Determine return indentation based on context
                # Look at previous non-empty line to determine nesting
                prev_line_idx = i - 1
                while prev_line_idx >= 0 and not lines[prev_line_idx].strip():
                    prev_line_idx -= 1
                
                if prev_line_idx >= 0:
                    prev_stripped = lines[prev_line_idx].strip()
                    if prev_stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:')):
                        fixed_lines.append('            ' + stripped + '\n')  # 12 spaces (nested)
                    else:
                        fixed_lines.append('        ' + stripped + '\n')  # 8 spaces (method level)
                else:
                    fixed_lines.append('        ' + stripped + '\n')  # 8 spaces
            
            elif stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', 'def ', 'class ')):
                # Control structures in method
                fixed_lines.append('        ' + stripped + '\n')  # 8 spaces
            
            elif stripped.startswith(('self.', 'super(', 'pass', 'continue', 'break', 'raise')):
                # Method body statements
                fixed_lines.append('        ' + stripped + '\n')  # 8 spaces
            
            else:
                # General method body content
                # Check if it's inside a control structure
                if any(keyword in lines[max(0, i-5):i] for keyword in ['if ', 'for ', 'while ', 'try:', 'with ']):
                    # Likely inside a control structure
                    fixed_lines.append('            ' + stripped + '\n')  # 12 spaces
                else:
                    fixed_lines.append('        ' + stripped + '\n')  # 8 spaces
            
            # Check if method ends (next line is a new method or class ends)
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped.startswith(('def ', 'class ', '@')) or \
                   (not next_stripped and i + 2 < len(lines) and lines[i + 2].strip().startswith(('def ', 'class '))):
                    in_method = False
                    current_method = None
            continue
        
        # Class-level content (not in method)
        if in_class and not in_method:
            # Class variables or class-level statements
            fixed_lines.append('    ' + stripped + '\n')  # 4 spaces
            continue
        
        # Module level content
        fixed_lines.append(stripped + '\n')
    
    # Write fixed file
    print(f"Writing fixed file to {output_file}...")
    with open(output_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed file written to: {output_file}")
    print("Please review the fixed file and test it.")
    
    # Summary
    class_count = sum(1 for line in fixed_lines if line.strip().startswith('class '))
    method_count = sum(1 for line in fixed_lines if line.strip().startswith('def '))
    
    print(f"\nSummary:")
    print(f"  - Classes found and fixed: {class_count}")
    print(f"  - Methods/functions found and fixed: {method_count}")
    print(f"  - Total lines processed: {len(fixed_lines)}")

def validate_fixed_file(filename):
    """Validate that the fixed file has proper Python syntax"""
    print(f"\nValidating {filename}...")
    
    try:
        with open(filename, 'r') as f:
            code = f.read()
        
        # Try to compile the code
        compile(code, filename, 'exec')
        print("✅ File has valid Python syntax!")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error found at line {e.lineno}: {e.msg}")
        print(f"   Problem: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error validating file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_all_indentation.py <input_file.py>")
        print("This will create a fixed version with '_fixed' suffix")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.py', '_fixed.py')
    
    print(f"🔧 Fixing indentation in {input_file}...")
    print("=" * 60)
    
    # Fix the file
    fix_entire_file_indentation(input_file, output_file)
    
    # Validate the fixed file
    print("=" * 60)
    if validate_fixed_file(output_file):
        print(f"\n✅ SUCCESS! The fixed file is ready: {output_file}")
        print("\nNext steps:")
        print("1. Back up your original file: cp unified_uel_ai_system.py unified_uel_ai_system_backup.py")
        print("2. Replace with fixed version: mv unified_uel_ai_system_fixed.py unified_uel_ai_system.py")
        print("3. Test the application: python unified_uel_ai_system.py")
    else:
        print(f"\n⚠️ The fixed file still has issues. Manual review needed.")
        print(f"Check around the reported line number in {output_file}")