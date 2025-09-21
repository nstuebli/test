for (int x = 0; x < dataxsdstruct::FRAMESIZE; ++x)
  {
    // Sea noise main antenna
    if (m_dSeaNoiseAmplitude > 0.0)
    {
      // Calculate mono data; write first data to NoiseBuffer
      dNoiseVal = m_pdSeaNoise[x];

      dNoiseVal *= NOISEPEAKFACTOR;  // Ueff -> Uss
      dNoiseVal *= m_dSeaNoiseAmplitude * m_dEnvNoiseAmplAdj;
      dNoiseVal *= dLvlCorect;

      if ((m_eNoiseType & NoiseType::eNOISETYPE_WithLowpass) == NoiseType::eNOISETYPE_WithLowpass)
        m_pdSeaNoise[x] = m_pSeaNoiseMainFilter->update(dNoiseVal);  // Lowpass filter
    }

memset(m_pdNarrowSamples, 0, sizeof(double) * m_iNoOfSamples);

  double  *pdLineFormArray;

  int     iDriftTableIdx;
  double  dCenterFreq       = 0.0;
  double  dBasicFreq        = 0.0;
  double  dSynchFrqJitPhase = 0.0;
  double  dSynchDriftPhase  = 0.0;
  bool    bNewLine          = false;

  if (m_LofarGen.dMainAmpl > 0.0 && m_dLofarLineMainAmplFactor > 0.0)
  {
    // Limited to 260dB
    if (m_LofarGen.dMainAmpl > 1.0e7)
    {
      m_LofarGen.dMainAmpl = 1.0e7;
    }

    for (int x = 0; x < MAXNARROWGENLINES; ++x)
    {
      bNewLine = false;

      if ((m_LofarLines[x].iType > 0) &&
          (m_LofarLines[x].dFrequency > 0.0))
      {
        // Check if a new line
        if (m_LofarGen.Line[x].bBusy == false)
        {
          bNewLine = true;
        }

        // The clean line
        if ((m_LofarLines[x].eiSignalForm >= dataxsdstruct::eSIGNALFORM_SINE) &&
            (m_LofarLines[x].eiSignalForm < dataxsdstruct::eSIGNALFORM_LAST))
        {
          pdLineFormArray = &m_dFormTable[m_LofarLines[x].eiSignalForm][0];
        }
        else
        {
          pdLineFormArray = &m_dFormTable[dataxsdstruct::eSIGNALFORM_SINE][0];
        }

        if (m_LofarLines[x].bFirst == true)
        {
          double  dRand = fWhiteNoiseSample();

          double  dTmpFreq = m_LofarLines[x].dFrequency * (double)m_LofarLines[x].iType;

          dCenterFreq = dTmpFreq + (dRand * m_LofarLines[x].dFreqJitter * dTmpFreq);

          dBasicFreq = dCenterFreq / (double)m_LofarLines[x].iType;

          dSynchFrqJitPhase = m_LofarGen.Line[x].dPhase;
          dSynchDriftPhase  = m_LofarGen.Line[x].dDriftPhase;

          if (bNewLine == true)
          {
            if (m_LofarLines[x].dFreqJitter > 0.001)
            {
              m_LofarGen.Line[x].dDriftPhase = fWhiteNoiseSample(0.0, double(MAXPASSIVFORMTABLESIZE - 1));
            }
            else
            {
              m_LofarGen.Line[x].dDriftPhase = 0.0;
              m_LofarGen.Line[x].dDriftIncr  = 0.0;
            }
          }

          dSynchDriftPhase  = m_LofarGen.Line[x].dDriftPhase;
        }
        else
        {
          dCenterFreq = dBasicFreq * (double)m_LofarLines[x].iType;

          if (m_LofarLines[x].bSynchronize == true)
          {
            m_LofarGen.Line[x].dPhase      = dSynchFrqJitPhase;
            m_LofarGen.Line[x].dDriftPhase = dSynchDriftPhase;

            m_LofarLines[x].bSynchronize = false;
          }
        }

        if ((m_LofarLines[x].Drift.dRunTime < 0.1) || // Define has no jitter
            (m_LofarLines[x].dFreqJitter <= 0.001))
        {
          m_LofarGen.Line[x].dNarrowActFreq = dCenterFreq;
          m_LofarGen.Line[x].dDriftIncr     = 0.0;
        }
        else
        {
          if ((m_LofarLines[x].Drift.eiSignalForm >= dataxsdstruct::eSIGNALFORM_SINE) &&
              (m_LofarLines[x].Drift.eiSignalForm < dataxsdstruct::eSIGNALFORM_LAST))
          {
            m_LofarGen.Line[x].dDriftIncr = m_dUpdateStep / (m_LofarLines[x].Drift.dRunTime / (double)MAXPASSIVFORMTABLESIZE);

            m_LofarGen.Line[x].dNarrowActFreq = dCenterFreq +
                                                dCenterFreq *
                                                m_LofarLines[x].Drift.dFreqFactor *
                                                m_dFormTable[m_LofarLines[x].Drift.eiSignalForm][m_LofarGen.Line[x].iDriftTableIdx];
          }
          else
          {
            m_LofarGen.Line[x].dNarrowActFreq = dCenterFreq;
          }
        }

        if (m_LofarGen.Line[x].dNarrowActFreq < 0.0)
        {
          m_LofarGen.Line[x].dNarrowActFreq = 0.0;
        }

        double  dPhaseIncr = m_LofarGen.Line[x].dNarrowActFreq * m_dIncrFactor;

        // Generate the signals
        for (int y = 0; y < m_iNoOfSamples; ++y)
        {
          m_pdNarrowSamples[y] += pdLineFormArray[(int)m_LofarGen.Line[x].dPhase] *
                                  m_dLofarLineMainAmplFactor * m_LofarLines[x].dAmplFactor *
                                  m_LofarGen.dMainAmpl;

          // Calculate next line phase
          m_LofarGen.Line[x].dPhase += dPhaseIncr;

          if (m_LofarGen.Line[x].dPhase >= (double)MAXPASSIVFORMTABLESIZE)
          {
            m_LofarGen.Line[x].dPhase -= (double)MAXPASSIVFORMTABLESIZE;
          }
        }

        if (m_LofarLines[x].Drift.dRunTime >= 0.1)  // There is drift
        {
          m_LofarGen.Line[x].dDriftPhase += m_LofarGen.Line[x].dDriftIncr;

          if ((int)m_LofarGen.Line[x].dDriftPhase >= MAXPASSIVFORMTABLESIZE)
          {
            m_LofarGen.Line[x].dDriftPhase -= MAXPASSIVFORMTABLESIZE;
          }

          iDriftTableIdx = (int)m_LofarGen.Line[x].dDriftPhase;

          // Calculate new index
          if (iDriftTableIdx != m_LofarGen.Line[x].iDriftTableIdx)
          {
            m_LofarGen.Line[x].iDriftTableIdx = iDriftTableIdx;
          }
        }

        m_LofarGen.Line[x].bBusy = true;
      }
      else
      {
        m_LofarGen.Line[x].bBusy = false;
      }
    }
  }
