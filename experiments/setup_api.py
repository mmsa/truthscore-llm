"""
Setup script for OpenAI API configuration.

Run this to check your OpenAI API setup before running experiments.
"""

import os
import sys

def check_openai_setup():
    """Check if OpenAI is properly configured."""
    print("Checking OpenAI API setup...")
    print("=" * 60)
    
    # Check if package is installed
    try:
        import openai
        print("✓ OpenAI package is installed")
        print(f"  Version: {openai.__version__}")
    except ImportError:
        print("✗ OpenAI package is NOT installed")
        print("  Install with: pip install openai")
        return False
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print(f"✓ OPENAI_API_KEY is set")
        print(f"  Key starts with: {api_key[:10]}...")
    else:
        print("✗ OPENAI_API_KEY is NOT set")
        print("  Set it with: export OPENAI_API_KEY='your-key-here'")
        print("  Or add to your .env file")
        return False
    
    # Test API call
    print("\nTesting API connection...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=10
        )
        print(f"✓ API connection successful")
        print(f"  Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False


if __name__ == "__main__":
    success = check_openai_setup()
    if success:
        print("\n" + "=" * 60)
        print("✓ Setup complete! You can now run experiments.")
        print("  Run: python -m experiments.run_experiment")
    else:
        print("\n" + "=" * 60)
        print("✗ Setup incomplete. Please fix the issues above.")
        sys.exit(1)

