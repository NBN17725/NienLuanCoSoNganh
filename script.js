// Đổi tên hàm từ checkHallucination -> check_hallucination
async function check_hallucination() {
    const answerInput = document.getElementById('answer');
    const answer = answerInput ? answerInput.value.trim() : '';
    const btn = document.getElementById('btn-check');

    if (!answer) {
        alert('Vui lòng nhập hoặc dán đoạn văn bản cần kiểm tra!');
        return;
    }

    const originalBtnHTML = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang phân tích...';
    btn.disabled = true;

    try {
        // Gọi API tương đối
        const res = await fetch('/api/check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ answer })
        });

        if (!res.ok) {
            throw new Error(`Lỗi máy chủ: ${res.status}`);
        }

        const data = await res.json();

        // Hiển thị khối kết quả
        const resultsContainer = document.getElementById('results');
        resultsContainer.style.display = 'block';

        // 1. Cập nhật Badge
        const badge = document.getElementById('status-badge');
        if (data.is_hallucinated) {
            badge.className = 'badge status-hallucinated';
            badge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> PHÁT HIỆN ẢO GIÁC';
        } else {
            badge.className = 'badge status-correct';
            badge.innerHTML = '<i class="fa-solid fa-circle-check"></i> CÂU TRẢ LỜI CHÍNH XÁC';
        }

        // 2. Cập nhật Ngữ cảnh
        if (data.context_info) {
            document.getElementById('ctx-title').innerText = data.context_info.title || 'Không có';
            document.getElementById('ctx-score').innerText = data.context_info.score !== undefined ? data.context_info.score : '--';
        }
        document.getElementById('ctx-text').innerText = data.context || 'Không tìm thấy ngữ cảnh đối soát phù hợp.';

        // 3. Cập nhật Điểm số & Thanh phần trăm
        const correctVal = data.scores && data.scores.correct ? data.scores.correct : 0;
        const hallucinatedVal = data.scores && data.scores.hallucinated ? data.scores.hallucinated : 0;

        const correctPercent = Math.round(correctVal > 1 ? correctVal : correctVal * 100);
        const hallucinatedPercent = Math.round(hallucinatedVal > 1 ? hallucinatedVal : hallucinatedVal * 100);

        document.getElementById('score-correct').innerText = `${correctPercent}%`;
        document.getElementById('fill-correct').style.width = `${correctPercent}%`;

        document.getElementById('score-hallucinated').innerText = `${hallucinatedPercent}%`;
        document.getElementById('fill-hallucinated').style.width = `${hallucinatedPercent}%`;

        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (err) {
        alert('Không thể kết nối tới Backend FastAPI! Vui lòng kiểm tra lại server Python.');
        console.error('Lỗi khi gọi API:', err);
    } finally {
        btn.innerHTML = originalBtnHTML;
        btn.disabled = false;
    }
}