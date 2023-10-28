/*
 *  Copyright (c) Streamax Tech. and its affiliates.
 *  Authors:
 *         (1) Wang, Yu  Oct 25, 2023
*/
#pragma once

#include <vector>
#include "base/kaldi-common.h"
#include "decoder/decodable-itf.h"

namespace tlg
{
    class DecodableMatrixScaled: public eesen::DecodableInterface
    {
        public:
            DecodableMatrixScaled(const float *likes,
                                  int T,
                                  int D,
                                  float scale): m_likes(likes),
                                                m_T(T),
                                                m_D(D),
                                                m_scale(scale) 
                                  { }

        public:

            virtual int32 NumFramesReady() const 
            { 
                return m_T; 
            }
        
            virtual bool IsLastFrame(int32 frame) const 
            {
                return ( frame == m_T - 1 );
            }
        
            /* frame就是帧ID, 从0开始， tid是CTC输出的classID, 0代表blank
            * 在构建TLG图的时候,0被<eps>占用, 而1之后才是CTC的class ID, 因此CTC的class ID都整体往后移了一位
            * 所以在取概率的时候要-1 
            */
            virtual eesen::BaseFloat LogLikelihood(eesen::int32 frame, eesen::int32 tid) 
            {
                return m_scale * m_likes[ frame * m_D + tid - 1 ];
            }

            // Indices are one-based!  This is for compatibility with OpenFst.
            virtual eesen::int32 NumIndices() const 
            { 
                return m_D; 
            }

        private:
            const float * m_likes;
            int m_T;
            int m_D;
            float m_scale;
            // KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMatrixScaled);
    };

} // namespace tlg