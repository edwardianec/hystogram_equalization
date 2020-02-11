public static UInt16[] ToUint16(byte[] pixelData, PixelFormat pixelFormat)
        {
            if (pixelData == null) throw new ArgumentNullException();

            int pixelOccupy = pixelFormat.PixelOccupy;

            UInt16[] result = new UInt16[8 * pixelData.Length / pixelOccupy];
            
            int bitCount = 0;
            UInt32 accumulataor = 0;
            UInt32 mask = (UInt32)((1 << pixelOccupy) - 1);

            for (int i = 0, ix = 0; i < pixelData.Length; i++)
            {
                if (bitCount > pixelOccupy)
                {
                    result[ix++] = (UInt16)(accumulataor & mask);
                    accumulataor >>= pixelOccupy;
                    bitCount -= pixelOccupy;
                }

                accumulataor |= ((UInt32)pixelData[i] << bitCount);
                bitCount += 8;
            }

            return result;
        }