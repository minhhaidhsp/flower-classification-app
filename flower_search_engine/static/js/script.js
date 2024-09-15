document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.getElementById('search-form');
    const resultsDiv = document.getElementById('results');
    const uploadForm = document.getElementById('upload-form');
    const uploadInfoDiv = document.getElementById('upload-info');
    const uploadResultsDiv = document.getElementById('upload-results');
    const uploadedImagePreview = document.getElementById('uploaded-image-preview');
    
    let currentPage = 1;
    const resultsPerPage = 21; // Số kết quả hiển thị mỗi trang
    const maxPageButtons = 5; // Số nút trang hiển thị cùng lúc

    // Xử lý form search
    searchForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(searchForm);

        $.ajax({
            url: '/search',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                resultsDiv.innerHTML = '';
                currentPage = 1;
                
                if (data.images.length > 0) {
                    paginateResults(data.images, resultsDiv);
                } else {
                    resultsDiv.innerHTML = '<p>No images found for the keyword.</p>';
                }
            },
            error: function(xhr, status, error) {
                console.error('Search failed:', error);
                resultsDiv.innerHTML = '<p>Search failed. Please try again.</p>';
            }
        });
    });

    // Xử lý form upload tương tự như search
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(uploadForm);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                uploadInfoDiv.innerHTML = `<p>${data.message}</p>`;
                uploadResultsDiv.innerHTML = '';
                currentPage = 1;
                
                // Hiển thị ảnh đã upload cạnh nút search
                if (data.uploaded_image) {
                    uploadedImagePreview.style.display = 'block'; // Hiển thị div chứa ảnh
                    uploadedImagePreview.innerHTML = ''; // Reset nội dung

                    const imgElement = document.createElement('img');
                    imgElement.src = `/static/${data.uploaded_image}`;
                    imgElement.alt = 'Uploaded Image';
                    imgElement.classList.add('uploaded-preview-image');
                    uploadedImagePreview.appendChild(imgElement);
                }
                

                if (data.images.length > 0) {
                    paginateResults(data.images, uploadResultsDiv);
                } else {
                    uploadResultsDiv.innerHTML = '<p>No images found after upload.</p>';
                }
            },
            error: function(xhr, status, error) {
                console.error('Upload failed:', error);
                uploadInfoDiv.innerHTML = '<p>Upload failed. Please try again.</p>';
            }
        });
    });

    // Hàm phân trang cho cả search và upload
    function paginateResults(images, container) {
        const totalPages = Math.ceil(images.length / resultsPerPage);

        function displayPage(page) {
            container.innerHTML = '';
            const start = (page - 1) * resultsPerPage;
            const end = start + resultsPerPage;
            const pageResults = images.slice(start, end);

            pageResults.forEach(function(image) {
                const imgElement = document.createElement('img');
                imgElement.src = `/static/${image}`;
                imgElement.alt = `Flower`;
                imgElement.classList.add('flower-image');

                const divElement = document.createElement('div');
                divElement.classList.add('image-item');
                divElement.appendChild(imgElement);
                
                container.appendChild(divElement);
            });

            displayPagination(page, totalPages, container);
        }

        function displayPagination(currentPage, totalPages, container) {
            const paginationDiv = document.createElement('div');
            paginationDiv.classList.add('pagination');

            if (currentPage > 1) {
                const prevButton = document.createElement('button');
                prevButton.textContent = 'Previous';
                prevButton.addEventListener('click', function() {
                    displayPage(currentPage - 1);
                });
                paginationDiv.appendChild(prevButton);
            }

            let startPage = Math.max(1, currentPage - Math.floor(maxPageButtons / 2));
            let endPage = Math.min(totalPages, startPage + maxPageButtons - 1);

            if (endPage - startPage + 1 < maxPageButtons) {
                startPage = Math.max(1, endPage - maxPageButtons + 1);
            }

            for (let i = startPage; i <= endPage; i++) {
                const pageButton = document.createElement('button');
                pageButton.textContent = i;
                if (i === currentPage) {
                    pageButton.classList.add('active');
                }
                pageButton.addEventListener('click', function() {
                    displayPage(i);
                });
                paginationDiv.appendChild(pageButton);
            }

            if (currentPage < totalPages) {
                const nextButton = document.createElement('button');
                nextButton.textContent = 'Next';
                nextButton.addEventListener('click', function() {
                    displayPage(currentPage + 1);
                });
                paginationDiv.appendChild(nextButton);
            }

            const oldPagination = container.querySelector('.pagination');
            if (oldPagination) {
                oldPagination.remove();
            }
            container.appendChild(paginationDiv);
        }

        displayPage(currentPage);
    }
});
