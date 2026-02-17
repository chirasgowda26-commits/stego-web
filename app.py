#!/usr/bin/env python3
"""
Enhanced Steganography System with Multi-Layer Security Defense
Based on working LSB code with added: AES Encryption, Advanced Obfuscation, Anti-Steganalysis
Academic Project - Advanced Implementation
"""

import streamlit as st
import hashlib
import io
import hmac
from typing import Optional, Tuple
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from scipy.stats import chisquare


class EnhancedStegoSystem:
    """Multi-layer defense steganography with encryption and anti-detection"""
    
    MAGIC_HEADER = b'ESTG'  # Enhanced STEGanography
    VERSION = 3
    
    def __init__(self):
        self.salt_size = 16
        self.iv_size = 16
        self.key_size = 32  # AES-256
        self.hmac_size = 32  # SHA-256 HMAC
    
    # ==================== LAYER 1: ENCRYPTION ====================
    
    def _derive_encryption_key(self, password: str, salt: bytes) -> bytes:
        """Derive AES encryption key from password using PBKDF2"""
        return PBKDF2(password, salt, dkLen=self.key_size, count=100000)
    
    def encrypt_message(self, message: str, password: str) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt message using AES-256-CBC
        Returns: (encrypted_data, salt, iv)
        """
        salt = get_random_bytes(self.salt_size)
        iv = get_random_bytes(self.iv_size)
        
        key = self._derive_encryption_key(password, salt)
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        message_bytes = message.encode('utf-8')
        padded_message = pad(message_bytes, AES.block_size)
        encrypted = cipher.encrypt(padded_message)
        
        return encrypted, salt, iv
    
    def decrypt_message(self, encrypted_data: bytes, password: str, 
                       salt: bytes, iv: bytes) -> Optional[str]:
        """Decrypt AES-256-CBC encrypted message"""
        try:
            key = self._derive_encryption_key(password, salt)
            cipher = AES.new(key, AES.MODE_CBC, iv)
            decrypted_padded = cipher.decrypt(encrypted_data)
            decrypted = unpad(decrypted_padded, AES.block_size)
            return decrypted.decode('utf-8')
        except Exception:
            return None
    
    # ==================== LAYER 2: AUTHENTICATION ====================
    
    def _generate_hmac(self, data: bytes, secret_key: str) -> bytes:
        """Generate HMAC for message authentication"""
        key_bytes = secret_key.encode('utf-8')
        return hmac.new(key_bytes, data, hashlib.sha256).digest()
    
    def _verify_hmac(self, data: bytes, received_hmac: bytes, secret_key: str) -> bool:
        """Verify HMAC to prevent tampering"""
        expected_hmac = self._generate_hmac(data, secret_key)
        return hmac.compare_digest(expected_hmac, received_hmac)
    
    # ==================== LAYER 3: OBFUSCATION ====================
    
    def _derive_seed(self, secret_key: str, password: str) -> int:
        """Generate deterministic seed from credentials"""
        combined = f"{secret_key}::{password}".encode('utf-8')
        hash_digest = hashlib.sha256(combined).digest()
        return int.from_bytes(hash_digest[:4], byteorder='big')
    
    def _generate_pixel_sequence(self, total_pixels: int, seed: int, 
                                 required_pixels: int) -> np.ndarray:
        """Generate pseudo-random pixel selection sequence"""
        rng = np.random.RandomState(seed)
        if required_pixels > total_pixels:
            raise ValueError(f"Message too large! Need {required_pixels} pixels, have {total_pixels}")
        return rng.choice(total_pixels, size=required_pixels, replace=False)
    
    def _add_adaptive_noise(self, image_array: np.ndarray, seed: int, 
                           embedded_pixels: np.ndarray) -> np.ndarray:
        """
        Add adaptive noise to balance LSB distribution across entire image
        This defeats Chi-square steganalysis by maintaining 50/50 ratio
        """
        rng = np.random.RandomState(seed + 99999)
        height, width, channels = image_array.shape
        flat_array = image_array.copy().flatten()
        
        # First, count current LSB distribution
        current_lsb = flat_array & 1
        zeros_count = np.sum(current_lsb == 0)
        ones_count = np.sum(current_lsb == 1)
        total = len(flat_array)
        
        # Calculate how many we need to balance to 50/50
        target = total // 2
        
        # Create mask of non-embedded pixels (safe to modify)
        all_indices = set(range(total))
        embedded_set = set(embedded_pixels)
        non_embedded = np.array(list(all_indices - embedded_set))
        
        # Determine which LSB value we need more of
        if zeros_count < ones_count:
            # Need more zeros, flip some ones to zeros in non-embedded pixels
            needed_flips = min((ones_count - zeros_count) // 2, len(non_embedded) // 2)
            
            # Find non-embedded pixels that currently have LSB=1
            candidates = []
            for idx in non_embedded:
                if flat_array[idx] & 1 == 1:
                    candidates.append(idx)
            
            if len(candidates) > 0:
                flip_count = min(needed_flips, len(candidates))
                flip_indices = rng.choice(candidates, size=flip_count, replace=False)
                flat_array[flip_indices] ^= 1  # Flip LSB
                
        elif ones_count < zeros_count:
            # Need more ones, flip some zeros to ones in non-embedded pixels
            needed_flips = min((zeros_count - ones_count) // 2, len(non_embedded) // 2)
            
            # Find non-embedded pixels that currently have LSB=0
            candidates = []
            for idx in non_embedded:
                if flat_array[idx] & 1 == 0:
                    candidates.append(idx)
            
            if len(candidates) > 0:
                flip_count = min(needed_flips, len(candidates))
                flip_indices = rng.choice(candidates, size=flip_count, replace=False)
                flat_array[flip_indices] ^= 1  # Flip LSB
        
        # Add small random noise to further mask patterns
        random_noise_count = len(non_embedded) // 10  # 10% random flips
        if len(non_embedded) > 0:
            random_indices = rng.choice(non_embedded, 
                                       size=min(random_noise_count, len(non_embedded)), 
                                       replace=False)
            for idx in random_indices:
                if rng.random() > 0.5:
                    flat_array[idx] ^= 1
        
        return flat_array.reshape(image_array.shape)
    
    # ==================== CORE STEGANOGRAPHY ====================
    
    def _message_to_bits(self, message: bytes) -> list:
        """Convert bytes to list of bits"""
        bits = []
        for byte in message:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def _bits_to_message(self, bits: list) -> bytes:
        """Convert list of bits back to bytes"""
        bytes_list = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_bits = bits[i:i+8]
                byte_val = sum(bit << (7 - j) for j, bit in enumerate(byte_bits))
                bytes_list.append(byte_val)
        return bytes(bytes_list)
    
    # ==================== ENCODING WITH MULTI-LAYER DEFENSE ====================
    
    def encode_message(self, cover_image: Image.Image, message: str, 
                      secret_key: str, password: str,
                      use_encryption: bool = True,
                      anti_steganalysis: bool = True) -> Tuple[Optional[Image.Image], bool, str]:
        """
        Multi-layer defense encoding:
        1. AES-256 Encryption
        2. HMAC Authentication
        3. LSB Embedding with cryptographic pixel selection
        4. Adaptive noise to defeat steganalysis
        """
        
        try:
            # Convert to RGB
            if cover_image.mode == 'RGBA':
                cover_image = cover_image.convert('RGB')
            
            img_array = np.array(cover_image)
            height, width, channels = img_array.shape
            total_pixels = height * width * channels
            
            # LAYER 1: Encryption
            if use_encryption:
                encrypted, salt, iv = self.encrypt_message(message, password)
                payload = salt + iv + encrypted
            else:
                payload = message.encode('utf-8')
            
            # LAYER 2: HMAC Authentication
            message_hmac = self._generate_hmac(payload, secret_key)
            payload_with_hmac = message_hmac + payload
            
            # Create header
            header = self.MAGIC_HEADER
            header += bytes([self.VERSION])
            header += bytes([1 if use_encryption else 0])  # Encryption flag
            header += len(payload_with_hmac).to_bytes(4, byteorder='big')
            
            # Combine header + payload
            all_data = header + payload_with_hmac
            data_bits = self._message_to_bits(all_data)
            required_pixels = len(data_bits)
            
            # LAYER 3: Cryptographic Pixel Selection
            seed = self._derive_seed(secret_key, password)
            pixel_indices = self._generate_pixel_sequence(total_pixels, seed, required_pixels)
            
            # Encode in LSB
            stego_array = img_array.copy()
            flat_array = stego_array.flatten()
            
            for i, pixel_idx in enumerate(pixel_indices):
                flat_array[pixel_idx] = (flat_array[pixel_idx] & 0xFE) | data_bits[i]
            
            stego_array = flat_array.reshape(img_array.shape)
            
            # LAYER 4: Anti-Steganalysis Defense
            if anti_steganalysis:
                stego_array = self._add_adaptive_noise(stego_array, seed, pixel_indices)
            
            stego_img = Image.fromarray(stego_array.astype(np.uint8))
            
            return stego_img, True, ""
            
        except Exception as e:
            return None, False, str(e)
    
    # ==================== DECODING ====================
    
    def decode_message(self, stego_image: Image.Image, secret_key: str, 
                      password: str) -> Tuple[Optional[str], bool, str]:
        """
        Multi-layer defense decoding with authentication
        """
        
        try:
            if stego_image.mode == 'RGBA':
                stego_image = stego_image.convert('RGB')
            
            img_array = np.array(stego_image).flatten()
            total_pixels = len(img_array)
            
            # Generate same pixel sequence
            seed = self._derive_seed(secret_key, password)
            
            # Extract header (10 bytes = 80 bits)
            header_size = 80
            try:
                header_indices = self._generate_pixel_sequence(total_pixels, seed, header_size)
            except ValueError:
                return None, False, "Invalid credentials or corrupted image"
            
            header_bits = [img_array[idx] & 1 for idx in header_indices]
            header_bytes = self._bits_to_message(header_bits)
            
            # Validate magic header
            if header_bytes[:4] != self.MAGIC_HEADER:
                return None, False, "Authentication failed: Invalid credentials"
            
            version = header_bytes[4]
            encryption_flag = header_bytes[5]
            payload_length = int.from_bytes(header_bytes[6:10], byteorder='big')
            
            # Extract payload
            total_bits = header_size + (payload_length * 8)
            
            try:
                all_indices = self._generate_pixel_sequence(total_pixels, seed, total_bits)
            except ValueError:
                return None, False, "Message extraction failed: Corrupted data"
            
            payload_bits = [img_array[idx] & 1 for idx in all_indices[header_size:]]
            payload_bytes = self._bits_to_message(payload_bits)
            
            # Verify HMAC (Layer 2)
            received_hmac = payload_bytes[:self.hmac_size]
            actual_payload = payload_bytes[self.hmac_size:]
            
            if not self._verify_hmac(actual_payload, received_hmac, secret_key):
                return None, False, "HMAC verification failed: Data may be tampered"
            
            # Decrypt if needed (Layer 1)
            if encryption_flag:
                salt = actual_payload[:self.salt_size]
                iv = actual_payload[self.salt_size:self.salt_size + self.iv_size]
                encrypted_data = actual_payload[self.salt_size + self.iv_size:]
                
                message = self.decrypt_message(encrypted_data, password, salt, iv)
                
                if message is None:
                    return None, False, "Decryption failed: Wrong password"
            else:
                message = actual_payload.decode('utf-8')
            
            return message, True, ""
            
        except Exception as e:
            return None, False, str(e)
    
    # ==================== BINARY VISUALIZATION MODULE ====================
    
    def message_to_binary_string(self, message: str) -> str:
        """
        Convert message to binary string representation for visualization
        Returns formatted binary string with 8-bit groupings
        """
        message_bytes = message.encode('utf-8')
        binary_groups = []
        
        for byte in message_bytes:
            binary_str = format(byte, '08b')
            binary_groups.append(binary_str)
        
        return ' '.join(binary_groups)
    
    def detect_steganography(self, image: Image.Image) -> Tuple[dict, bool, str]:
        """
        Chi-square steganalysis to detect hidden data
        Our anti-steganalysis defense should make this show CLEAN
        """
        
        try:
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            img_array = np.array(image).flatten()
            
            # Analyze LSB distribution
            lsb_values = img_array & 1
            
            zeros = np.sum(lsb_values == 0)
            ones = np.sum(lsb_values == 1)
            total = len(lsb_values)
            
            # Expected 50/50 distribution
            expected = total / 2
            observed = np.array([zeros, ones])
            expected_freq = np.array([expected, expected])
            
            # Chi-square test
            chi2_stat, p_value = chisquare(observed, expected_freq)
            
            results = {
                'total_pixels': total,
                'lsb_zeros': int(zeros),
                'lsb_ones': int(ones),
                'zero_ratio': zeros / total,
                'one_ratio': ones / total,
                'chi_square': float(chi2_stat),
                'p_value': float(p_value)
            }
            
            # Interpretation (with anti-steganalysis, should show CLEAN)
            if p_value < 0.05:
                results['verdict'] = "SUSPICIOUS"
                results['confidence'] = "High"
                results['explanation'] = "LSB distribution is statistically abnormal"
            elif p_value < 0.20:
                results['verdict'] = "POSSIBLY MODIFIED"
                results['confidence'] = "Medium"
                results['explanation'] = "LSB distribution shows minor anomalies"
            else:
                results['verdict'] = "CLEAN"
                results['confidence'] = "High"
                results['explanation'] = "LSB distribution appears natural"
            
            return results, True, ""
            
        except Exception as e:
            return {}, False, str(e)


def display_binary_visualization(stego: EnhancedStegoSystem, message: str):
    """Display only binary bits"""
    st.markdown("### 🔢 Binary Representation")
    
    # Show only binary string
    binary_string = stego.message_to_binary_string(message)
    st.code(binary_string, language=None)
    
    # Statistics only
    total_bits = len(message) * 8
    st.caption(f"Total: {total_bits} bits")


# ==================== STREAMLIT UI ====================

def main():
    """Streamlit application"""
    
    st.set_page_config(
        page_title="Image Based Steganography Tool",
        page_icon="🔒",
        layout="wide"
    )
    
    st.title("🔒 IMAGE BASED STEGANOGRAPHY TOOL")
    st.markdown("""
    Hide and extract secret messages within images securely.
    """)
    
    stego = EnhancedStegoSystem()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Encode", "🔍 Decode", "🔬 Steganalysis", "ℹ️ How to Use"])
    
    # ==================== ENCODE TAB ====================
    with tab1:
        st.header("Hide Secret Message with Multi-Layer Defense")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. Upload Cover Image")
            uploaded = st.file_uploader("Cover Image (PNG recommended)", 
                                       type=['png', 'jpg', 'jpeg'], key="enc_img")
            
            if uploaded:
                cover_img = Image.open(uploaded)
                st.image(cover_img, caption="Cover Image", use_container_width=True)
        
        with col2:
            st.subheader("2. Message & Security Settings")
            
            message = st.text_area("Secret Message", height=100,
                                  placeholder="Enter your confidential message...")
            
            st.markdown("**Security Options:**")
            use_encryption = st.checkbox("🔐 Enable AES-256 Encryption", value=True,
                                        help="Encrypts message for additional security")
            anti_steganalysis = st.checkbox("🎭 Enable Anti-Detection Defense", value=True,
                                           help="Makes hidden message harder to detect")
            
            secret_key = st.text_input("Secret Key", type="default",
                                      placeholder="Enter secret key")
            password = st.text_input("Password", type="password",
                                    placeholder="Enter password")
            
            st.markdown("---")
            encode_btn = st.button("🔐 Hide Message", 
                                  type="primary", use_container_width=True)
        
        if encode_btn:
            if not uploaded or not message or not secret_key or not password:
                st.error("❌ Please fill all fields!")
            else:
                with st.spinner("Hiding message..."):
                    result, success, error = stego.encode_message(
                        cover_img, message, secret_key, password,
                        use_encryption, anti_steganalysis
                    )
                
                if success and result:
                    st.success("### 🎉 Message Successfully Hidden!")
                    
                    # Display binary visualization after successful encoding
                    st.markdown("---")
                    display_binary_visualization(stego, message)
                    st.markdown("---")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(result, caption="Stego Image (Indistinguishable)", 
                                use_container_width=True)
                    with col_b:
                        buf = io.BytesIO()
                        result.save(buf, format='PNG')
                        st.download_button(
                            "📥 Download Stego Image",
                            buf.getvalue(),
                            "stego_defense.png",
                            "image/png",
                            use_container_width=True
                        )
                        
                        st.warning("⚠️ **Critical**: Save your credentials securely!\n\n"
                                  "You need both Secret Key AND Password to extract the message later.")
                else:
                    st.error(f"❌ Encoding failed: {error}")
    
    # ==================== DECODE TAB ====================
    with tab2:
        st.header("Extract Hidden Message")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. Upload Stego Image")
            stego_file = st.file_uploader("Stego Image", 
                                         type=['png', 'jpg', 'jpeg'], key="dec_img")
            
            if stego_file:
                stego_img = Image.open(stego_file)
                st.image(stego_img, caption="Stego Image", use_container_width=True)
        
        with col2:
            st.subheader("2. Enter Credentials")
            
            dec_key = st.text_input("Secret Key", type="default", 
                                   placeholder="Enter the secret key", key="dec_key")
            dec_pass = st.text_input("Password", type="password",
                                    placeholder="Enter the password", key="dec_pass")
            
            st.info("🔑 Both credentials must match exactly as used during encoding")
            
            st.markdown("---")
            decode_btn = st.button("🔓 Extract Message", type="primary", 
                                  use_container_width=True)
        
        if decode_btn:
            if not stego_file or not dec_key or not dec_pass:
                st.error("❌ Please fill all fields!")
            else:
                with st.spinner("Extracting message..."):
                    msg, success, error = stego.decode_message(stego_img, dec_key, dec_pass)
                
                if success and msg:
                    st.success("### 🎉 Message Successfully Extracted!")
                    st.markdown("---")
                    st.subheader("🔓 Decoded Message:")
                    st.info(msg)
                    st.code(msg, language=None)
                else:
                    st.error(f"❌ {error}")
    
    # ==================== STEGANALYSIS TAB ====================
    with tab3:
        st.header("🔬 Detection Test")
        st.markdown("Test if hidden data can be detected in images")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image to Analyze")
            analysis_file = st.file_uploader("Test Image", 
                                            type=['png', 'jpg', 'jpeg'], key="analysis_img")
            
            if analysis_file:
                analysis_img = Image.open(analysis_file)
                st.image(analysis_img, caption="Image for Analysis", use_container_width=True)
                
                analyze_btn = st.button("🔬 Run Chi-Square Test", 
                                       type="primary", use_container_width=True)
                
                if analyze_btn:
                    with st.spinner("Running steganalysis..."):
                        results, success, error = stego.detect_steganography(analysis_img)
                    
                    if not success:
                        st.error(f"❌ Analysis error: {error}")
        
        with col2:
            if analysis_file and 'analyze_btn' in locals() and analyze_btn and results:
                st.subheader("Analysis Results")
                
                verdict_color = {
                    "CLEAN": "🟢",
                    "POSSIBLY MODIFIED": "🟡",
                    "SUSPICIOUS": "🔴"
                }
                
                st.markdown(f"### {verdict_color.get(results['verdict'], '⚪')} {results['verdict']}")
                st.markdown(f"**Confidence:** {results['confidence']}")
                st.markdown(f"**Explanation:** {results['explanation']}")
                
                st.markdown("---")
                
                st.markdown("**Statistical Analysis:**")
                st.metric("Chi-Square Statistic", f"{results['chi_square']:.4f}")
                st.metric("P-Value", f"{results['p_value']:.4f}")
                
                st.markdown("**LSB Distribution:**")
                col_a, col_b = st.columns(2)
                col_a.metric("LSB = 0", f"{results['lsb_zeros']:,}")
                col_b.metric("LSB = 1", f"{results['lsb_ones']:,}")
                
                st.progress(results['one_ratio'])
                st.caption(f"Ratio: {results['zero_ratio']:.3f} / {results['one_ratio']:.3f}")
                
                if results['verdict'] == "CLEAN":
                    st.success("✅ **Detection Defense Working!**\n\n"
                              "The image appears clean.")
                
                st.info("""
                **Result Meaning:**
                - CLEAN: No obvious hidden data detected
                - POSSIBLY MODIFIED: May contain hidden data
                - SUSPICIOUS: Likely contains hidden data
                """)
    
    # ==================== INFO TAB ====================
    with tab4:
        st.header("ℹ️ How to Use")
        
        st.markdown("""
        ### 📝 Encode
        1. Upload image  
        2. Enter message  
        3. Enter secret key & password  
        4. Click *Hide Message*  
        5. Download stego image  
        
        ### 🔍 Decode
        1. Upload stego image  
        2. Enter same key & password  
        3. Click *Extract Message*
        
        ### 🔬 Steganalysis
        1. Upload any image
        2. Click *Run Chi-Square Test*
        3. View detection results
        
        ---
        
        ### 🛡️ Security Features
        
        **Four-Layer Protection:**
        - 🔐 AES-256 Encryption
        - 🔏 HMAC Authentication
        - 🎲 Cryptographic Pixel Selection
        - 🎭 Anti-Detection Defense
        
        ---
        
        ### ⚠️ Important Notes
        
        ✅ **Always use PNG format** (lossless)  
        ✅ **Save credentials securely**  
        ✅ **Test extraction immediately**  
        
        ❌ **Avoid JPEG** (lossy compression)  
        ❌ **Don't resize or edit** the stego image  
        ❌ **Don't share credentials** insecurely  
        
        ---
        
        ### 📚 Educational Purpose Only
        
        This tool is designed for learning about:
        - Steganography techniques
        - Cryptographic security
        - Information hiding
        - Digital forensics
        
        **Not for highly classified information.**
        """)


if __name__ == "__main__":
    main()